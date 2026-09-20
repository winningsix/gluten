# Q4 Flux development snapshot

Development branch consolidating the existing local Flux integration work:
driver coordination and lifecycle, native ORC writes, projection/shuffle
planning, scan/cache configuration, and targeted regression tests. Based on
`4366ebbc8`, with the existing remote range-stream and Spark 4 task-context
fixes retained by merge. No old development branch is force-pushed.

Pair with `winningsix/velox-1` branch `Q4`. This source snapshot includes
experiments beyond the isolated linked runtime used for the historical FINRA
measurements. It must not be presented as a fully rebuilt, performance-qualified
release. A clean paired build and runtime regression remain prerequisites for
promotion. No performance defaults are promoted as part of the cleanup.

Historical qualification: the local parallel-H2D candidate's exact JAR was
SHA-256 `0bbdab8ced7ed4faeb109d91df24f60a1ae914f584384ff9e44189c8f07ea92b`.
This does not establish that every current source modification was in that JAR.

See [tools/q4](tools/q4/README.md) for the offline time-series replay and its
data provenance, limitations and regeneration tests. EFA/TCP measurements are
not fabricated from local evidence. No AWS resources/tickets are created.
