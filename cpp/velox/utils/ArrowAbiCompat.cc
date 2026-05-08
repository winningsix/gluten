/*
 * Compatibility shims for Velox archives built against older Arrow headers.
 *
 * The runtime container provides Arrow 18, where a few APIs changed from
 * const-reference/string signatures to value/string_view signatures. Some
 * prebuilt Velox objects still reference the old mangled symbols, so expose
 * those symbols here and forward them to the current Arrow API.
 */

#include <arrow/array/array_binary.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

extern "C" int arrowKeyValueMetadataFindKeyString(
    const arrow::KeyValueMetadata* metadata,
    const std::string& key)
    asm("_ZNK5arrow16KeyValueMetadata7FindKeyERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");

extern "C" int arrowKeyValueMetadataFindKeyString(
    const arrow::KeyValueMetadata* metadata,
    const std::string& key) {
  return metadata->FindKey(key);
}

extern "C" const uint8_t* arrowFixedSizeBinaryArrayGetValue(
    const arrow::FixedSizeBinaryArray* array,
    int64_t index) asm("_ZNK5arrow20FixedSizeBinaryArray8GetValueEl");

extern "C" const uint8_t* arrowFixedSizeBinaryArrayGetValue(
    const arrow::FixedSizeBinaryArray* array,
    int64_t index) {
  return array->GetValue(index);
}

extern "C" std::shared_ptr<arrow::DataType> arrowListField(
    const std::shared_ptr<arrow::Field>& valueType)
    asm("_ZN5arrow4listERKSt10shared_ptrINS_5FieldEE");

extern "C" std::shared_ptr<arrow::DataType> arrowListField(
    const std::shared_ptr<arrow::Field>& valueType) {
  return std::make_shared<arrow::ListType>(valueType);
}

extern "C" std::shared_ptr<arrow::DataType> arrowLargeListField(
    const std::shared_ptr<arrow::Field>& valueType)
    asm("_ZN5arrow10large_listERKSt10shared_ptrINS_5FieldEE");

extern "C" std::shared_ptr<arrow::DataType> arrowLargeListField(
    const std::shared_ptr<arrow::Field>& valueType) {
  return std::make_shared<arrow::LargeListType>(valueType);
}

extern "C" std::shared_ptr<arrow::DataType> arrowFixedSizeListField(
    const std::shared_ptr<arrow::Field>& valueType,
    int listSize) asm("_ZN5arrow15fixed_size_listERKSt10shared_ptrINS_5FieldEEi");

extern "C" std::shared_ptr<arrow::DataType> arrowFixedSizeListField(
    const std::shared_ptr<arrow::Field>& valueType,
    int listSize) {
  return std::make_shared<arrow::FixedSizeListType>(valueType, listSize);
}
