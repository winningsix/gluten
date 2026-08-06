/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "JniFileSystem.h"
#include "jni/JniCommon.h"
#include "velox/common/io/IoStatistics.h"

#include <cstring>
#include <limits>
#include <vector>

extern "C" JNIEXPORT void JNICALL Java_com_nvidia_sparkmpp_CrtS3RangeReader_copyDirect(
    JNIEnv* env,
    jclass,
    jlong destinationAddress,
    jobject source,
    jint sourceOffset,
    jint length);

extern "C" JNIEXPORT void JNICALL Java_com_nvidia_sparkmpp_CrtS3RangeReader_copyArray(
    JNIEnv* env,
    jclass,
    jlong destinationAddress,
    jbyteArray source,
    jint sourceOffset,
    jint length);

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_sparkmpp_CrtS3RangeReader_wrapDirect(JNIEnv* env, jclass, jlong destinationAddress, jlong length);

namespace {
constexpr std::string_view kJniFsScheme("jni:");
constexpr std::string_view kJolFsScheme("jol:");

JavaVM* vm;

jclass jniFileSystemClass;
jclass jniReadFileClass;
jclass jniWriteFileClass;
jclass crtS3RangeReaderClass;

jmethodID jniGetFileSystem;
jmethodID jniIsCapableForNewFile;
jmethodID jniFileSystemOpenFileForRead;
jmethodID jniFileSystemOpenFileForWrite;
jmethodID jniFileSystemRemove;
jmethodID jniFileSystemRename;
jmethodID jniFileSystemExists;
jmethodID jniFileSystemList;
jmethodID jniFileSystemMkdir;
jmethodID jniFileSystemRmdir;

jmethodID jniReadFilePread;
jmethodID jniReadFileShouldCoalesce;
jmethodID jniReadFileSize;
jmethodID jniReadFileMemoryUsage;
jmethodID jniReadFileGetNaturalReadSize;
jmethodID jniReadFileClose;

jmethodID jniWriteFileAppend;
jmethodID jniWriteFileFlush;
jmethodID jniWriteFileClose;
jmethodID jniWriteFileSize;
jmethodID crtS3ObjectSize;
jmethodID crtS3ReadRanges;

jstring createJString(JNIEnv* env, const std::string_view& path) {
  return env->NewStringUTF(std::string(path).c_str());
}

std::string_view removePathSchema(std::string_view path) {
  unsigned long pos = path.find(':');
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

class JniReadFile : public facebook::velox::ReadFile {
 public:
  explicit JniReadFile(jobject obj) {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    obj_ = env->NewGlobalRef(obj);
    checkException(env);
  }

  ~JniReadFile() override {
    try {
      closeInternal();
      JNIEnv* env = nullptr;
      attachCurrentThreadAsDaemonOrThrow(vm, &env);
      env->DeleteGlobalRef(obj_);
      checkException(env);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Error closing jni read file " << e.what();
    }
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const facebook::velox::FileIoContext& fileStorageContext = {}) const override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    env->CallVoidMethod(
        obj_, jniReadFilePread, static_cast<jlong>(offset), static_cast<jlong>(length), reinterpret_cast<jlong>(buf));
    checkException(env);
    return std::string_view(reinterpret_cast<const char*>(buf));
  }

  bool shouldCoalesce() const override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    jboolean out = env->CallBooleanMethod(obj_, jniReadFileShouldCoalesce);
    checkException(env);
    return out;
  }

  uint64_t size() const override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    jlong out = env->CallLongMethod(obj_, jniReadFileSize);
    checkException(env);
    return static_cast<uint64_t>(out);
  }

  uint64_t memoryUsage() const override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    jlong out = env->CallLongMethod(obj_, jniReadFileMemoryUsage);
    checkException(env);
    return static_cast<uint64_t>(out);
  }

  std::string getName() const override {
    return "<JniReadFile>";
  }

  uint64_t getNaturalReadSize() const override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    jlong out = env->CallLongMethod(obj_, jniReadFileGetNaturalReadSize);
    checkException(env);
    return static_cast<uint64_t>(out);
  }

 private:
  void closeInternal() {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    env->CallVoidMethod(obj_, jniReadFileClose);
    checkException(env);
  }

  jobject obj_;
};

class JniWriteFile : public facebook::velox::WriteFile {
 public:
  explicit JniWriteFile(jobject obj) {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    obj_ = env->NewGlobalRef(obj);
    checkException(env);
  }

  ~JniWriteFile() override {
    try {
      closeInternal();
      JNIEnv* env = nullptr;
      attachCurrentThreadAsDaemonOrThrow(vm, &env);
      env->DeleteGlobalRef(obj_);
      checkException(env);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Error closing jni write file " << e.what();
    }
  }

  void append(std::string_view data) override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    const void* bytes = data.data();
    unsigned long len = data.size();
    env->CallVoidMethod(obj_, jniWriteFileAppend, static_cast<jlong>(len), reinterpret_cast<jlong>(bytes));
    checkException(env);
  }

  void flush() override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    env->CallVoidMethod(obj_, jniWriteFileFlush);
    checkException(env);
  }

  void close() override {
    closeInternal();
  }

  uint64_t size() const override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    jlong out = env->CallLongMethod(obj_, jniWriteFileSize);
    checkException(env);
    return static_cast<uint64_t>(out);
  }

 private:
  void closeInternal() {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    env->CallVoidMethod(obj_, jniWriteFileClose);
    checkException(env);
  }

  jobject obj_;
};

// Convert "xxx:/a/b/c" to "/a/b/c". Probably it's Velox's job to remove the protocol when calling the member
// functions?
class FileSystemWrapper : public facebook::velox::filesystems::FileSystem {
 public:
  static std::shared_ptr<facebook::velox::filesystems::FileSystem> wrap(
      std::shared_ptr<facebook::velox::filesystems::FileSystem> fs) {
    return std::shared_ptr<facebook::velox::filesystems::FileSystem>(new FileSystemWrapper(fs));
  }

  std::string name() const override {
    return fs_->name();
  }

  std::unique_ptr<facebook::velox::ReadFile> openFileForRead(
      std::string_view path,
      const facebook::velox::filesystems::FileOptions& options) override {
    return fs_->openFileForRead(rewrite(path), options);
  }

  std::unique_ptr<facebook::velox::WriteFile> openFileForWrite(
      std::string_view path,
      const facebook::velox::filesystems::FileOptions& options) override {
    return fs_->openFileForWrite(rewrite(path), options);
  }

  void remove(std::string_view path) override {
    fs_->remove(rewrite(path));
  }

  void rename(std::string_view oldPath, std::string_view newPath, bool overwrite) override {
    fs_->rename(rewrite(oldPath), rewrite(newPath), overwrite);
  }

  bool exists(std::string_view path) override {
    return fs_->exists(rewrite(path));
  }

  std::vector<std::string> list(std::string_view path) override {
    return fs_->list(rewrite(path));
  }

  void mkdir(std::string_view path, const facebook::velox::filesystems::DirectoryOptions& options = {}) override {
    fs_->mkdir(rewrite(path));
  }

  void rmdir(std::string_view path) override {
    fs_->rmdir(rewrite(path));
  }

 private:
  FileSystemWrapper(std::shared_ptr<facebook::velox::filesystems::FileSystem> fs) : FileSystem({}), fs_(fs) {}

  static std::string_view rewrite(std::string_view path) {
    return removePathSchema(path);
  }

  std::shared_ptr<facebook::velox::filesystems::FileSystem> fs_;
};

class JniFileSystem : public facebook::velox::filesystems::FileSystem {
 public:
  explicit JniFileSystem(jobject obj, std::shared_ptr<const facebook::velox::config::ConfigBase> config)
      : FileSystem(config) {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    obj_ = env->NewGlobalRef(obj);
    checkException(env);
  }

  ~JniFileSystem() override {
    try {
      JNIEnv* env = nullptr;
      attachCurrentThreadAsDaemonOrThrow(vm, &env);
      env->DeleteGlobalRef(obj_);
      checkException(env);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Error releasing jni file system " << e.what();
    }
  }

  std::string name() const override {
    return "JNI FS";
  }

  std::unique_ptr<facebook::velox::ReadFile> openFileForRead(
      std::string_view path,
      const facebook::velox::filesystems::FileOptions& options) override {
    GLUTEN_CHECK(
        options.values.empty(),
        "JniFileSystem::openFileForRead: file options is not empty, this is not currently supported");
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    jobject obj = env->CallObjectMethod(obj_, jniFileSystemOpenFileForRead, createJString(env, path));
    checkException(env);
    auto out = std::make_unique<JniReadFile>(obj);
    return out;
  }

  std::unique_ptr<facebook::velox::WriteFile> openFileForWrite(
      std::string_view path,
      const facebook::velox::filesystems::FileOptions& options) override {
    GLUTEN_CHECK(
        options.values.empty(),
        "JniFileSystem::openFileForWrite: file options is not empty, this is not currently supported");
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    jobject obj = env->CallObjectMethod(obj_, jniFileSystemOpenFileForWrite, createJString(env, path));
    checkException(env);
    auto out = std::make_unique<JniWriteFile>(obj);
    return out;
  }

  void remove(std::string_view path) override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    env->CallVoidMethod(obj_, jniFileSystemRemove, createJString(env, path));
    checkException(env);
  }

  void rename(std::string_view oldPath, std::string_view newPath, bool overwrite) override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    env->CallVoidMethod(obj_, jniFileSystemRename, createJString(env, oldPath), createJString(env, newPath), overwrite);
    checkException(env);
  }

  bool exists(std::string_view path) override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    bool out = env->CallBooleanMethod(obj_, jniFileSystemExists, createJString(env, path));
    checkException(env);
    return out;
  }

  std::vector<std::string> list(std::string_view path) override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    std::vector<std::string> out;
    jobjectArray jarray =
        static_cast<jobjectArray>(env->CallObjectMethod(obj_, jniFileSystemList, createJString(env, path)));
    checkException(env);
    jsize length = env->GetArrayLength(jarray);
    out.reserve(length);
    for (jsize i = 0; i < length; ++i) {
      jstring element = static_cast<jstring>(env->GetObjectArrayElement(jarray, i));
      std::string cElement = jStringToCString(env, element);
      out.push_back(cElement);
      env->DeleteLocalRef(element);
    }
    return out;
  }

  void mkdir(std::string_view path, const facebook::velox::filesystems::DirectoryOptions& options = {}) override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    env->CallVoidMethod(obj_, jniFileSystemMkdir, createJString(env, path));
    checkException(env);
  }

  void rmdir(std::string_view path) override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    env->CallVoidMethod(obj_, jniFileSystemRmdir, createJString(env, path));
    checkException(env);
  }

  static bool isCapableForNewFile(uint64_t size) {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);
    bool out = env->CallStaticBooleanMethod(jniFileSystemClass, jniIsCapableForNewFile, static_cast<jlong>(size));
    checkException(env);
    return out;
  }

  static std::function<bool(std::string_view)> schemeMatcher() {
    return [](std::string_view filePath) { return filePath.find(kJniFsScheme) == 0; };
  }

  static std::function<
      std::shared_ptr<FileSystem>(std::shared_ptr<const facebook::velox::config::ConfigBase>, std::string_view)>
  fileSystemGenerator() {
    return [](std::shared_ptr<const facebook::velox::config::ConfigBase> properties, std::string_view filePath) {
      JNIEnv* env = nullptr;
      attachCurrentThreadAsDaemonOrThrow(vm, &env);
      jobject obj = env->CallStaticObjectMethod(jniFileSystemClass, jniGetFileSystem);
      checkException(env);
      // remove "jni:" or "jol:" prefix.
      std::shared_ptr<FileSystem> lfs = FileSystemWrapper::wrap(std::make_shared<JniFileSystem>(obj, properties));
      return lfs;
    };
  }

 private:
  jobject obj_;
};
} // namespace

void gluten::initVeloxJniFileSystem(JNIEnv* env) {
  // vm
  if (env->GetJavaVM(&vm) != JNI_OK) {
    throw gluten::GlutenException("Unable to get JavaVM instance");
  }

  // classes
  jniFileSystemClass = createGlobalClassReferenceOrError(env, "Lorg/apache/gluten/fs/JniFilesystem;");
  jniReadFileClass = createGlobalClassReferenceOrError(env, "Lorg/apache/gluten/fs/JniFilesystem$ReadFile;");
  jniWriteFileClass = createGlobalClassReferenceOrError(env, "Lorg/apache/gluten/fs/JniFilesystem$WriteFile;");

  // methods in JniFilesystem
  jniGetFileSystem =
      getStaticMethodIdOrError(env, jniFileSystemClass, "getFileSystem", "()Lorg/apache/gluten/fs/JniFilesystem;");
  jniIsCapableForNewFile = getStaticMethodIdOrError(env, jniFileSystemClass, "isCapableForNewFile", "(J)Z");
  jniFileSystemOpenFileForRead = getMethodIdOrError(
      env, jniFileSystemClass, "openFileForRead", "(Ljava/lang/String;)Lorg/apache/gluten/fs/JniFilesystem$ReadFile;");
  jniFileSystemOpenFileForWrite = getMethodIdOrError(
      env,
      jniFileSystemClass,
      "openFileForWrite",
      "(Ljava/lang/String;)Lorg/apache/gluten/fs/JniFilesystem$WriteFile;");
  jniFileSystemRemove = getMethodIdOrError(env, jniFileSystemClass, "remove", "(Ljava/lang/String;)V");
  jniFileSystemRename =
      getMethodIdOrError(env, jniFileSystemClass, "rename", "(Ljava/lang/String;Ljava/lang/String;Z)V");
  jniFileSystemExists = getMethodIdOrError(env, jniFileSystemClass, "exists", "(Ljava/lang/String;)Z");
  jniFileSystemList = getMethodIdOrError(env, jniFileSystemClass, "list", "(Ljava/lang/String;)[Ljava/lang/String;");
  jniFileSystemMkdir = getMethodIdOrError(env, jniFileSystemClass, "mkdir", "(Ljava/lang/String;)V");
  jniFileSystemRmdir = getMethodIdOrError(env, jniFileSystemClass, "rmdir", "(Ljava/lang/String;)V");

  // methods in JniFilesystem$ReadFile
  jniReadFilePread = getMethodIdOrError(env, jniReadFileClass, "pread", "(JJJ)V");
  jniReadFileShouldCoalesce = getMethodIdOrError(env, jniReadFileClass, "shouldCoalesce", "()Z");
  jniReadFileSize = getMethodIdOrError(env, jniReadFileClass, "size", "()J");
  jniReadFileMemoryUsage = getMethodIdOrError(env, jniReadFileClass, "memoryUsage", "()J");
  jniReadFileGetNaturalReadSize = getMethodIdOrError(env, jniReadFileClass, "getNaturalReadSize", "()J");
  jniReadFileClose = getMethodIdOrError(env, jniReadFileClass, "close", "()V");

  // methods in JniFilesystem$WriteFile
  jniWriteFileAppend = getMethodIdOrError(env, jniWriteFileClass, "append", "(JJ)V");
  jniWriteFileFlush = getMethodIdOrError(env, jniWriteFileClass, "flush", "()V");
  jniWriteFileClose = getMethodIdOrError(env, jniWriteFileClass, "close", "()V");
  jniWriteFileSize = getMethodIdOrError(env, jniWriteFileClass, "size", "()J");

  // This class is supplied only by runtimes that enable the Java AWS CRT
  // bridge. Keep it optional: the native C++ S3 CRT path does not need it.
  crtS3RangeReaderClass = createGlobalClassReference(env, "Lcom/nvidia/sparkmpp/CrtS3RangeReader;");
  if (crtS3RangeReaderClass != nullptr) {
    JNINativeMethod nativeMethods[] = {
        {const_cast<char*>("copyDirect"),
         const_cast<char*>("(JLjava/nio/ByteBuffer;II)V"),
         reinterpret_cast<void*>(Java_com_nvidia_sparkmpp_CrtS3RangeReader_copyDirect)},
        {const_cast<char*>("copyArray"),
         const_cast<char*>("(J[BII)V"),
         reinterpret_cast<void*>(Java_com_nvidia_sparkmpp_CrtS3RangeReader_copyArray)},
        {const_cast<char*>("wrapDirect"),
         const_cast<char*>("(JJ)Ljava/nio/ByteBuffer;"),
         reinterpret_cast<void*>(Java_com_nvidia_sparkmpp_CrtS3RangeReader_wrapDirect)}};
    const auto registerResult =
        env->RegisterNatives(crtS3RangeReaderClass, nativeMethods, sizeof(nativeMethods) / sizeof(nativeMethods[0]));
    if (registerResult != JNI_OK) {
      checkException(env);
    }
    GLUTEN_CHECK(registerResult == JNI_OK, "Failed to register AWS CRT S3 range bridge native methods");
    crtS3ObjectSize = getStaticMethodIdOrError(env, crtS3RangeReaderClass, "objectSize", "(Ljava/lang/String;)J");
    crtS3ReadRanges =
        getStaticMethodIdOrError(env, crtS3RangeReaderClass, "readRanges", "(Ljava/lang/String;J[J[J[J)J");
    LOG(INFO) << "AWS CRT S3 range bridge is available";
  }
}

void gluten::finalizeVeloxJniFileSystem(JNIEnv* env) {
  if (crtS3RangeReaderClass != nullptr) {
    env->DeleteGlobalRef(crtS3RangeReaderClass);
    crtS3RangeReaderClass = nullptr;
  }
  env->DeleteGlobalRef(jniWriteFileClass);
  env->DeleteGlobalRef(jniReadFileClass);
  env->DeleteGlobalRef(jniFileSystemClass);

  vm = nullptr;
}

// "jol" stands for letting Gluten choose between jni fs and local fs.
// This doesn't implement facebook::velox::filesystems::FileSystem since it just
// act as a entry-side router to create JniFilesystem and LocalFilesystem
void gluten::registerJolFileSystem(uint64_t maxFileSize) {
  GLUTEN_CHECK(maxFileSize > 0, "Unexpected max file size for jol fs: " + std::to_string(maxFileSize));

  auto JolSchemeMatcher = [](std::string_view filePath) { return filePath.find(kJolFsScheme) == 0; };

  auto fileSystemGenerator =
      [maxFileSize](
          std::shared_ptr<const facebook::velox::config::ConfigBase> properties,
          std::string_view filePath) -> std::shared_ptr<facebook::velox::filesystems::FileSystem> {
    // select JNI file if there is enough space
    if (JniFileSystem::isCapableForNewFile(maxFileSize)) {
      return JniFileSystem::fileSystemGenerator()(properties, filePath);
    }

    // otherwise select local file
    // remove "jol:" to make Velox choose local fs.
    auto localFilePath = removePathSchema(filePath);
    auto fs = FileSystemWrapper::wrap(facebook::velox::filesystems::getFileSystem(localFilePath, properties));
    return fs;
  };

  facebook::velox::filesystems::registerFileSystem(JolSchemeMatcher, fileSystemGenerator);
}

extern "C" bool glutenCrtS3RangeReaderAvailable() {
  return crtS3RangeReaderClass != nullptr;
}

extern "C" uint64_t glutenCrtS3ObjectSize(const char* uri) {
  GLUTEN_CHECK(crtS3RangeReaderClass != nullptr, "AWS CRT S3 range bridge is not available");
  JNIEnv* env = nullptr;
  attachCurrentThreadAsDaemonOrThrow(vm, &env);
  auto jUri = env->NewStringUTF(uri);
  checkException(env);
  const auto size = env->CallStaticLongMethod(crtS3RangeReaderClass, crtS3ObjectSize, jUri);
  env->DeleteLocalRef(jUri);
  checkException(env);
  GLUTEN_CHECK(size >= 0, "AWS CRT S3 object size is negative");
  return static_cast<uint64_t>(size);
}

extern "C" uint64_t glutenCrtS3ReadRanges(
    const char* uri,
    uint8_t* destination,
    const uint64_t* offsets,
    const uint64_t* lengths,
    const uint64_t* destinationOffsets,
    size_t count) {
  GLUTEN_CHECK(crtS3RangeReaderClass != nullptr, "AWS CRT S3 range bridge is not available");
  GLUTEN_CHECK(count <= static_cast<size_t>(std::numeric_limits<jsize>::max()), "Too many AWS CRT S3 ranges");

  JNIEnv* env = nullptr;
  attachCurrentThreadAsDaemonOrThrow(vm, &env);
  const auto jCount = static_cast<jsize>(count);
  auto jUri = env->NewStringUTF(uri);
  auto jOffsets = env->NewLongArray(jCount);
  auto jLengths = env->NewLongArray(jCount);
  auto jDestinationOffsets = env->NewLongArray(jCount);
  checkException(env);

  std::vector<jlong> signedOffsets(count);
  std::vector<jlong> signedLengths(count);
  std::vector<jlong> signedDestinationOffsets(count);
  for (size_t index = 0; index < count; ++index) {
    GLUTEN_CHECK(
        offsets[index] <= static_cast<uint64_t>(std::numeric_limits<jlong>::max()),
        "AWS CRT S3 range offset exceeds jlong");
    GLUTEN_CHECK(
        lengths[index] <= static_cast<uint64_t>(std::numeric_limits<jlong>::max()),
        "AWS CRT S3 range length exceeds jlong");
    GLUTEN_CHECK(
        destinationOffsets[index] <= static_cast<uint64_t>(std::numeric_limits<jlong>::max()),
        "AWS CRT S3 destination offset exceeds jlong");
    signedOffsets[index] = static_cast<jlong>(offsets[index]);
    signedLengths[index] = static_cast<jlong>(lengths[index]);
    signedDestinationOffsets[index] = static_cast<jlong>(destinationOffsets[index]);
  }
  env->SetLongArrayRegion(jOffsets, 0, jCount, signedOffsets.data());
  env->SetLongArrayRegion(jLengths, 0, jCount, signedLengths.data());
  env->SetLongArrayRegion(jDestinationOffsets, 0, jCount, signedDestinationOffsets.data());
  checkException(env);

  const auto bytes = env->CallStaticLongMethod(
      crtS3RangeReaderClass,
      crtS3ReadRanges,
      jUri,
      reinterpret_cast<jlong>(destination),
      jOffsets,
      jLengths,
      jDestinationOffsets);
  env->DeleteLocalRef(jDestinationOffsets);
  env->DeleteLocalRef(jLengths);
  env->DeleteLocalRef(jOffsets);
  env->DeleteLocalRef(jUri);
  checkException(env);
  GLUTEN_CHECK(bytes >= 0, "AWS CRT S3 read byte count is negative");
  return static_cast<uint64_t>(bytes);
}

extern "C" JNIEXPORT void JNICALL Java_com_nvidia_sparkmpp_CrtS3RangeReader_copyDirect(
    JNIEnv* env,
    jclass,
    jlong destinationAddress,
    jobject source,
    jint sourceOffset,
    jint length) {
  auto* sourceAddress = static_cast<uint8_t*>(env->GetDirectBufferAddress(source));
  const auto capacity = env->GetDirectBufferCapacity(source);
  if (sourceAddress == nullptr || sourceOffset < 0 || length < 0 ||
      static_cast<jlong>(sourceOffset) + length > capacity) {
    jclass errorClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(errorClass, "Invalid direct CRT response buffer");
    return;
  }
  std::memcpy(reinterpret_cast<void*>(destinationAddress), sourceAddress + sourceOffset, static_cast<size_t>(length));
}

extern "C" JNIEXPORT void JNICALL Java_com_nvidia_sparkmpp_CrtS3RangeReader_copyArray(
    JNIEnv* env,
    jclass,
    jlong destinationAddress,
    jbyteArray source,
    jint sourceOffset,
    jint length) {
  const auto arrayLength = env->GetArrayLength(source);
  if (sourceOffset < 0 || length < 0 || static_cast<jlong>(sourceOffset) + length > arrayLength) {
    jclass errorClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(errorClass, "Invalid heap CRT response buffer");
    return;
  }
  env->GetByteArrayRegion(source, sourceOffset, length, reinterpret_cast<jbyte*>(destinationAddress));
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_sparkmpp_CrtS3RangeReader_wrapDirect(JNIEnv* env, jclass, jlong destinationAddress, jlong length) {
  if (destinationAddress == 0 || length < 0 || length > std::numeric_limits<jint>::max()) {
    jclass errorClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(errorClass, "Invalid direct CRT destination buffer");
    return nullptr;
  }
  return env->NewDirectByteBuffer(reinterpret_cast<void*>(destinationAddress), length);
}
