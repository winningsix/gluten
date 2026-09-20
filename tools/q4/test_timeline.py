import json
from pathlib import Path
import tempfile
import unittest

from build_timeline import bin_samples, finite, load_spark_stages, write_rates


class TimelineTest(unittest.TestCase):
    def test_missing_gpu_is_not_zero(self):
        self.assertEqual(bin_samples([(10, 0, 80)], 0, 250000000), [[0.0, None]])

    def test_equal_gpu_weight_not_sample_count(self):
        self.assertEqual(bin_samples([(10, 0, 20), (20, 0, 20), (30, 1, 80)],
                                     0, 250000000), [[0.0, 50.0]])

    def test_invalid_fields(self):
        for value in ('N/A', 'NaN', 'inf', '-1', '101'):
            self.assertIsNone(finite(value, 100))

    def test_only_leaf_no_tail(self):
        rows = [dict(epoch_ns=t, **{'io.stat': f'252:0 wbytes={b}\n259:2 wbytes={b}'})
                for t, b in [(0, 0), (1000000000, 2000000000), (2000000000, 6000000000)]]
        self.assertEqual(write_rates(rows, 0, 1500000000, '259:2'), [[0.5, 2.0]])

    def test_counter_reset_unavailable(self):
        rows = [dict(epoch_ns=t, **{'io.stat': f'259:2 wbytes={b}'})
                for t, b in [(0, 20), (1000000000, 10)]]
        self.assertEqual(write_rates(rows, 0, 1000000000, '259:2'), [[0.5, None]])

    def test_stage_intersection_and_role(self):
        events = [
            {'Event': 'SparkListenerStageSubmitted', 'Stage Info': {
                'Stage ID': 1, 'Stage Attempt ID': 0, 'Stage Name': 'RDD',
                'Number of Tasks': 4, 'Submission Time': 900}},
            {'Event': 'SparkListenerStageCompleted', 'Stage Info': {
                'Stage ID': 1, 'Stage Attempt ID': 0, 'Stage Name': 'RDD',
                'Number of Tasks': 4, 'Submission Time': 900, 'Completion Time': 1200}},
            {'Event': 'SparkListenerStageCompleted', 'Stage Info': {
                'Stage ID': 9, 'Stage Attempt ID': 0, 'Stage Name': 'outside',
                'Number of Tasks': 1, 'Submission Time': 100, 'Completion Time': 200}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'eventlog'
            path.write_text('\n'.join(json.dumps(event) for event in events))
            stages = load_spark_stages(path, 1_000_000_000, 1_500_000_000, 'Flux')
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]['label'], 'Spark wrapper — native pipeline')
        self.assertEqual((stages[0]['start_s'], stages[0]['end_s']), (0, 0.2))
        self.assertTrue(stages[0]['clipped_start'])


if __name__ == '__main__':
    unittest.main()
