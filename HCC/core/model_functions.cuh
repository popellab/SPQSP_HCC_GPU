#ifndef FLAMEGPU_MODEL_FUNCTIONS_CUH
#define FLAMEGPU_MODEL_FUNCTION_CUH

#include "flamegpu/flamegpu.h"
#include "../qsp/LymphCentral_wrapper.h"

namespace HCC {

    void set_internal_params(flamegpu::ModelDescription& model, const HCC::LymphCentralWrapper& lymph);

}

#endif