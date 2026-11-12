/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/boosting.h>

#include <memory>
#include <string>

#include "dart.hpp"
#include "gbdt.h"
#include "rf.hpp"

#ifdef USE_MULTI_GPU
#include "cuda/nccl_gbdt.hpp"
#endif  // USE_MULTI_GPU

namespace LightGBM {

std::string GetBoostingTypeFromModelFile(const char* filename) {
  TextReader<size_t> model_reader(filename, true);
  std::string type = model_reader.first_line();
  return type;
}

bool Boosting::LoadFileToBoosting(Boosting* boosting, const char* filename) {
  auto start_time = std::chrono::steady_clock::now();
  if (boosting != nullptr) {
    TextReader<size_t> model_reader(filename, true);
    size_t buffer_len = 0;
    auto buffer = model_reader.ReadContent(&buffer_len);
    if (!boosting->LoadModelFromString(buffer.data(), buffer_len)) {
      return false;
    }
  }
  std::chrono::duration<double, std::milli> delta = (std::chrono::steady_clock::now() - start_time);
  Log::Debug("Time for loading model: %f seconds", 1e-3*delta);
  return true;
}

Boosting* Boosting::CreateBoosting(const std::string& type, const char* filename,
  const std::string&
  #ifdef USE_CUDA
  device_type
  #endif  // USE_CUDA
  , const int
  #ifdef USE_CUDA
  num_gpu
  #endif  // USE_CUDA
  ) {
  #if defined(USE_CUDA) && !defined(USE_MULTI_GPU)
  if (type == std::string("gbdt") && device_type == std::string("cuda") && num_gpu > 1) {
    Log::Fatal("Multi-GPU training is not available in this LightGBM build. "
               "Rebuild with a supported GPU backend and -DENABLE_MULTI_GPU=ON.");
  }
  #endif  // defined(USE_CUDA) && !defined(USE_MULTI_GPU)
  if (filename == nullptr || filename[0] == '\0') {
    if (type == std::string("gbdt")) {
      #ifdef USE_MULTI_GPU
      if (device_type == std::string("cuda") && num_gpu > 1) {
        return new NCCLGBDT<GBDT>();
      } else {
      #endif  // USE_MULTI_GPU
        return new GBDT();
      #ifdef USE_MULTI_GPU
      }
      #endif  // USE_MULTI_GPU
    } else if (type == std::string("dart")) {
      return new DART();
    } else if (type == std::string("goss")) {
      return new GBDT();
    } else if (type == std::string("rf")) {
      return new RF();
    } else {
      return nullptr;
    }
  } else {
    std::unique_ptr<Boosting> ret;
    if (GetBoostingTypeFromModelFile(filename) == std::string("tree")) {
      if (type == std::string("gbdt")) {
        #ifdef USE_MULTI_GPU
        if (device_type == std::string("cuda") && num_gpu > 1) {
          ret.reset(new NCCLGBDT<GBDT>());
        } else {
        #endif  // USE_MULTI_GPU
          ret.reset(new GBDT());
        #ifdef USE_MULTI_GPU
        }
        #endif  // USE_MULTI_GPU
      } else if (type == std::string("dart")) {
        ret.reset(new DART());
      } else if (type == std::string("goss")) {
        ret.reset(new GBDT());
      } else if (type == std::string("rf")) {
        ret.reset(new RF());
      } else {
        Log::Fatal("Unknown boosting type %s", type.c_str());
      }
      LoadFileToBoosting(ret.get(), filename);
    } else {
      Log::Fatal("Unknown model format or submodel type in model file %s", filename);
    }
    return ret.release();
  }
}

}  // namespace LightGBM
