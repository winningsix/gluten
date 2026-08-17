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

// Minimal build-only consumer proving that the installed AWS SDK exposes both
// S3 implementations, Velox's assume-role provider, and their linkable static
// dependency closure.

#include <aws/core/Aws.h>
#include <aws/identity-management/auth/STSAssumeRoleCredentialsProvider.h>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3/S3Client.h>

int main() {
  Aws::SDKOptions options;
  Aws::InitAPI(options);
  {
    Aws::S3::S3Client s3;
    Aws::S3Crt::S3CrtClient s3Crt;
    Aws::Auth::STSAssumeRoleCredentialsProvider assumeRole(
        "arn:aws:iam::000000000000:role/gluten-carrier-probe",
        "gluten-carrier-probe");
  }
  Aws::ShutdownAPI(options);
  return 0;
}
