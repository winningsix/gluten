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
package org.apache.gluten.vectorized;

public class ShufflePayloadCatalogJniWrapper {

  // Returns null if block not in catalog (not found).
  // Returns empty array if found but empty partition.
  // Returns array of DirectByteBuffers pointing to
  // native catalog memory (zero-copy).
  public static native java.nio.ByteBuffer[] resolveBlockDirect(
      int shuffleId, long mapId, int partitionId);

  public static native void unregisterShuffle(int shuffleId);

  public static native void unregisterMapOutput(int shuffleId, long mapId);
}
