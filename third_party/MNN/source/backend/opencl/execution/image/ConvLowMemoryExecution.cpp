//  ConvLowMemoryExecution.cpp
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_LOW_MEMORY
#include "ConvLowMemoryExecution.hpp"
// #define LOG_VERBOSE
namespace MNN {
namespace OpenCL {

// set mDequantScale mDequantOffset mNumQuantBit mFilterDataPtr from mConv2dParams
void ConvLowMemoryExecution::getInfoFromOpLowMemory(std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    quanCommon = ConvolutionCommon::load(mConv2dParams, this->backend(), false, true);
    if ((mOpenCLBackend->getMemory() == BackendConfig::Memory_Low) && (mConv2dParams->quanParameter() != nullptr)) {
        mLowMemoryFlag = true;
    } else {
        MNN_ERROR("Conv buf low memory init error.\n");
        MNN_ASSERT(false);
    }
    // set mNumQuantBit
    if (quanCommon->quan->type() == 4) {
        mNumQuantBit = 8;
    } else if (quanCommon->quan->type() == 1 || quanCommon->quan->type() == 2) {
        mNumQuantBit = 4;
    } else {/* More types to be supported. */}
    // src of alpha in CPU
    float * dequantAlpha = quanCommon->alpha.get();
    int numAlpha = mOutputChannel;
    // set mDequantScale mDequantOffset
    int numAlphaPack = ROUND_UP(numAlpha, 16);
    int numBiasPack = ROUND_UP(mOutputChannel, 16);
    int bytes = mOpenCLBackend->fpBytes();
    mResource->biasBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ROUND_UP(mOutputChannel, 16) * bytes));
    mResource->dequantScaleBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, numAlphaPack * bytes));
    mResource->dequantOffsetBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, numAlphaPack * bytes));
    // transfer data from src in cpu to dst in gpu
    cl_int resBias, resScale, resOffset;
    auto biasPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->biasBuffer.get()), true, CL_MAP_WRITE, 0, numBiasPack * bytes, nullptr, nullptr, &resBias);
    void * dequantScaleBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->dequantScaleBuffer.get()), true, CL_MAP_WRITE, 0, numAlphaPack * bytes, nullptr, nullptr, &resScale);
    void * dequantOffsetBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->dequantOffsetBuffer.get()), true, CL_MAP_WRITE, 0, numAlphaPack * bytes, nullptr, nullptr, &resOffset);

    if (biasPtrCL != nullptr && resBias == CL_SUCCESS) {
        ::memset(biasPtrCL, 0, numBiasPack * bytes);
        if (nullptr != mConv2dParams->bias()) {
            const float *biasDataPtr = mConv2dParams->bias()->data();
            if (bytes == 2){
                for(int i = 0; i < mOutputChannel; i++) {
                    ((half_float::half*)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
                }
            } else {
                ::memcpy(biasPtrCL, biasDataPtr, mOutputChannel * sizeof(float));
            }
        }
    }
    ::memset(dequantScaleBufferMap, -1, numAlphaPack * bytes);
    ::memset(dequantOffsetBufferMap, 0, numAlphaPack * bytes);
    if (dequantScaleBufferMap != nullptr && dequantOffsetBufferMap != nullptr && resScale == CL_SUCCESS && resOffset == CL_SUCCESS) {
        if (bytes == 2) {
            if (quanCommon->asymmetric) {
                for (int i = 0; i < numAlpha; ++i) {
                    ((half_float::half *)dequantOffsetBufferMap)[i] = (half_float::half)dequantAlpha[2 * i];
                    ((half_float::half *)dequantScaleBufferMap)[i] = (half_float::half)dequantAlpha[2 * i + 1];
                }
            } else {
                for (int i = 0; i < numAlpha; ++i) {
                    ((half_float::half *)dequantScaleBufferMap)[i] = (half_float::half)dequantAlpha[i];
                    ((half_float::half *)dequantOffsetBufferMap)[i] = 0.0f;
                }
            }
        } else {
            if (quanCommon->asymmetric) {
                for (int i = 0; i < numAlpha; ++i) {
                    ((float *)dequantOffsetBufferMap)[i] = dequantAlpha[2 * i];
                    ((float *)dequantScaleBufferMap)[i] = dequantAlpha[2 * i + 1];
                }
            } else {
                for (int i = 0; i < numAlpha; ++i) {
                    ((float *)dequantScaleBufferMap)[i] = dequantAlpha[i];
                    ((float *)dequantOffsetBufferMap)[i] = 0.0f;
                }
            }
        }
    } else {
        MNN_ERROR("Map error dequantBufferMap == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->biasBuffer.get()), biasPtrCL);
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->dequantScaleBuffer.get()), dequantScaleBufferMap);
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->dequantOffsetBuffer.get()), dequantOffsetBufferMap);
    // set mFilterDataPtr
    mFilterDataPtr = (void *)quanCommon->weight.get();
}
// set mKernelBuffer for the 1x1 kernels
void ConvLowMemoryExecution::set1x1WeightLowMemory(int packCout, int packCin, void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    cl_int res;
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mOutputChannel, 8)/*Cout pack set to max 8*/, ROUND_UP(mInputChannel, packCin), mKernelWidth, mKernelHeight}));
    size_t buffer_size = filterBuffer->usize() / sizeof(float);
    float *dequantAlpha = quanCommon->alpha.get();
    // shared part for all cases
    if (mNumQuantBit == 8) {
        // int8 case
        buffer_size *= sizeof(int8_t);
    } else if (mNumQuantBit == 4){
        // int4 case
        buffer_size /= 2;
    } else {/* More types to be supported. */}
    mResource->kernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->kernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
    if(kernelBufferPtr != nullptr && res == CL_SUCCESS){
        ::memset(kernelBufferPtr, 0, buffer_size);

        if(mResource->gemmOpt){
            for(int o = 0; o < mOutputChannel; o++){
                float zero = 0;
                if(quanCommon->asymmetric){
                    zero = (-dequantAlpha[2 * o + 1])/dequantAlpha[2 * o];
                }
                int i = 0;
                for(; i < mInputChannel; i++){
                    int bufferIdx = (o/packCout) * packCin*packCout + (i/packCin)*packCin*ROUND_UP(mOutputChannel, packCout) + (i%packCin)*packCout + (o%packCout);//(Ci/packCin， Co/packCout, packCin， packCout)
                    int filterIdx = o*mInputChannel + i;
                    if (mNumQuantBit == 8) {
                        // int8 case
                        ((int8_t *)kernelBufferPtr)[bufferIdx] = (int8_t)(((int8_t *)filterDataPtr)[filterIdx]);
                    } else if (mNumQuantBit == 4){
                        // int4 case
                        if (bufferIdx % 2 == 0) {
                            ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)((((int8_t *)filterDataPtr)[filterIdx] + 8) * 16);
                        } else {
                            ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)(((int8_t *)filterDataPtr)[filterIdx] + 8);
                        }
                    } else {/* More types to be supported. */}
                }
                for(; i < ROUND_UP(mInputChannel, 4); i++){
                    int bufferIdx = (o/packCout) * packCin*packCout + (i/packCin)*packCin*ROUND_UP(mOutputChannel, packCout) + (i%packCin)*packCout + (o%packCout);//(Ci/packCin， Co/packCout, packCin， packCout)
                    if (mNumQuantBit == 8) {
                        // int8 case
                        ((int8_t *)kernelBufferPtr)[bufferIdx] = (int8_t)(zero);
                    } else if (mNumQuantBit == 4){
                        // int4 case
                        if (bufferIdx % 2 == 0) {
                            ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)((zero + 8) * 16);
                        } else {
                            ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)(zero + 8);
                        }
                    }
                }
            }
        }else{
            for(int o = 0; o < mOutputChannel; o++){
                float zero = 0;
                if(quanCommon->asymmetric){
                    zero = (-dequantAlpha[2 * o + 1])/dequantAlpha[2 * o];
                }
                int i = 0;
                for(; i < mInputChannel; i++){
                    int bufferIdx = (o/packCout) * ROUND_UP(mInputChannel, packCin)*packCout + (i/packCin)*packCin*packCout + (o%packCout) + (i%packCin)*packCout;//(Co/packCout, Ci/packCin, packCin, packCout)
                    int filterIdx = o*mInputChannel + i;
                    if (mNumQuantBit == 8) {
                        // int8 case
                        ((int8_t *)kernelBufferPtr)[bufferIdx] = (int8_t)(((int8_t *)filterDataPtr)[filterIdx]);
                    } else if (mNumQuantBit == 4){
                        // int4 case
                        if (bufferIdx % 2 == 0) {
                            ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)((((int8_t *)filterDataPtr)[filterIdx] + 8) * 16);
                        } else {
                            ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)(((int8_t *)filterDataPtr)[filterIdx] + 8);
                        }
                    } else {/* More types to be supported. */}
                }
                for(; i < ROUND_UP(mInputChannel, 4); i++){
                    int bufferIdx = (o/packCout) * ROUND_UP(mInputChannel, packCin)*packCout + (i/packCin)*packCin*packCout + (o%packCout)*packCin + (i%packCin);//(Co/packCout, Ci/packCin, packCout, packCin)
                    if (mNumQuantBit == 8) {
                        // int8 case
                        ((int8_t *)kernelBufferPtr)[bufferIdx] = (int8_t)(zero);
                    } else if (mNumQuantBit == 4){
                        // int4 case
                        if (bufferIdx % 2 == 0) {
                            ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)((zero + 8) * 16);
                        } else {
                            ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)(zero + 8);
                        }
                    }
                }
            }
        }
    } else {
        MNN_ERROR("set1x1WeightLowMemory: Map error ptrCL == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->kernelBuffer.get()), kernelBufferPtr);
}
// set mFilter for the general kernels
void ConvLowMemoryExecution::setGeneralWeightLowMemory(void* filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    if (filterDataPtr != nullptr) {
        std::vector<int> filterImageShape{ROUND_UP(mInputChannel, 4), (UP_DIV(mOutputChannel, 4) * mKernelWidth * mKernelHeight)};
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({mOutputChannel, ROUND_UP(mInputChannel, 4), mKernelWidth, mKernelHeight}));
        // int buffer_size = filterBuffer->elementSize();
        size_t buffer_size = filterBuffer->usize() / sizeof(float);
        buffer_size *= sizeof(int8_t);
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
        float *dequantAlpha = quanCommon->alpha.get();
        // map and pack data from filterDataPtr
        cl_int res;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if(ptrCL != nullptr && res == CL_SUCCESS) {
            ::memset(ptrCL, 0, buffer_size);
            const int copy_size = mKernelWidth * mKernelHeight * sizeof(int8_t);
            for(int oc=0; oc<mOutputChannel; oc++) {
                float zero = 0;
                if(quanCommon->asymmetric){
                    zero = (-dequantAlpha[2 * oc + 1])/dequantAlpha[2 * oc];
                }
                int ic = 0;
                for(; ic<mInputChannel; ic++) {
                    ::memcpy((int8_t *)ptrCL + (oc * ROUND_UP(mInputChannel, 4) + ic) * mKernelWidth * mKernelHeight, ((int8_t *)filterDataPtr) + (oc * mInputChannel + ic) * mKernelWidth * mKernelHeight, copy_size);
                }
                for(; ic<ROUND_UP(mInputChannel, 4); ic++) {
                    ((int8_t *)ptrCL)[(oc * ROUND_UP(mInputChannel, 4) + ic) * mKernelWidth * mKernelHeight] = (int8_t)(zero);
                }
            }
        } else {
            MNN_ERROR("setGeneralWeightLowMemory: Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
        // convert to NC4HW4
        if (mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->filter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
            mResource->kernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
            mResource->filter->buffer().device = (uint64_t)(mResource->kernelBuffer.get());
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->filter.get(), false, true, mLowMemoryFlag, mNumQuantBit);
        } else if (mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->filter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 2 * filterImageShape[0]}));
            mResource->kernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size/2));
            mResource->filter->buffer().device = (uint64_t)(mResource->kernelBuffer.get());
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->filter.get(), false, true, mLowMemoryFlag, mNumQuantBit);
        } else {/* More types to be supported. */}
    } else {
        MNN_ERROR("GetConvParams Error: filterDataPtr == nullptr. \n");
        MNN_ASSERT(false);
    }
}
// select the fastest kernel for the 1x1 cases by tuning
void ConvLowMemoryExecution::tune1x1CaseLowMemory(Tensor * input, Tensor * output) {
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mOpenCLBackend->startRecord(mRecording);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(mKernelHeight) + "_" + std::to_string(mKernelWidth) + "_" + std::to_string(mStrides[0]) + "_" + std::to_string(mStrides[1]) + "_" + std::to_string(mDilations[0]) + "_" + std::to_string(mDilations[1]);
    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {height, width};
    int stideShape[2]       = {mStrides[0], mStrides[1]};
    const int total_kernel = 2;
    std::string kernelName[total_kernel] = {"conv_2d_1x1", "conv_2d_1x1_c8h1w4"};
    int itemC[total_kernel] = {4, 8};
    int itemH[total_kernel] = {1, 1};
    int itemW[total_kernel] = {4, 4};
    int actual_kernel = total_kernel;

    cl::Kernel kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    cl_int ret = CL_SUCCESS;
    for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        std::set<std::string> buildOption = mResource->buildOptions;
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[knl_idx], buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
        
        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
        uint32_t idx            = 0;
        ret |= kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx].setArg(idx++, openCLImage(input));
        ret |= kernel[knl_idx].setArg(idx++, *mResource->kernelBuffer.get());
        ret |= kernel[knl_idx].setArg(idx++, *mResource->dequantScaleBuffer.get());
        ret |= kernel[knl_idx].setArg(idx++, *mResource->dequantOffsetBuffer.get());
        ret |= kernel[knl_idx].setArg(idx++, *mResource->biasBuffer.get());
        ret |= kernel[knl_idx].setArg(idx++, openCLImage(output));
        ret |= kernel[knl_idx].setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= kernel[knl_idx].setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= kernel[knl_idx].setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(stideShape), stideShape);
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(width, 4));
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(outputShape.at(3), 4));
        
        std::pair<std::vector<uint32_t>, uint32_t> retTune;
        retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
        
        //printf("conv1x1 kernel_%d = %d  [%d, %d]\n", knl_idx, retTune.second, retTune.first[0], retTune.first[1]);
        if(min_cost.first > retTune.second) {
            min_cost.first = retTune.second;
            min_cost.second = knl_idx;
            mLocalWorkSize = {retTune.first[0], retTune.first[1]};
        }
    }

    int min_index  = min_cost.second;
    mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
    std::set<std::string> buildOption = mResource->buildOptions;
    mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[min_index], buildOption);
    uint32_t idx = 0;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, openCLImage(input));
    ret |= mKernel.setArg(idx++, *mResource->kernelBuffer.get());
    ret |= mKernel.setArg(idx++, *mResource->dequantScaleBuffer.get());
    ret |= mKernel.setArg(idx++, *mResource->dequantOffsetBuffer.get());
    ret |= mKernel.setArg(idx++, *mResource->biasBuffer.get());
    ret |= mKernel.setArg(idx++, openCLImage(output));
    ret |= mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
    ret |= mKernel.setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= mKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= mKernel.setArg(idx++, sizeof(stideShape), stideShape);
    ret |= mKernel.setArg(idx++, UP_DIV(width, 4));
    ret |= mKernel.setArg(idx++, UP_DIV(outputShape.at(3), 4));
    MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1LowMemory");
    mOpenCLBackend->recordKernel2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
    mOpenCLBackend->endRecord(mRecording);
    return;
}
// select the fastest kernel for the general cases by tuning
void ConvLowMemoryExecution::tuneGeneralCaseLowMemory(Tensor * input, Tensor * output) {
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mOpenCLBackend->startRecord(mRecording);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(mKernelHeight) + "_" + std::to_string(mKernelWidth) + "_" + std::to_string(mStrides[0]) + "_" + std::to_string(mStrides[1]) + "_" + std::to_string(mDilations[0]) + "_" + std::to_string(mDilations[1]);
    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {height, width};
    int kernelShape[2]      = {mKernelHeight, mKernelWidth};
    int strideShape[2]      = {mStrides[0], mStrides[1]};
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int dilationShape[2]    = {mDilations[0], mDilations[1]};
    const int total_kernel = 3;
    std::string kernelName[total_kernel] = {"conv_2d_c4h1w4", "conv_2d_c4h4w1", "conv_2d_c8h4w1" };
    int itemC[total_kernel] = {4, 4, 8};
    int itemH[total_kernel] = {1, 4, 4};
    int itemW[total_kernel] = {4, 1, 1};
    int actual_kernel = total_kernel;
    cl::Kernel kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    // MNN_PRINT("Checking kernel %d.\n", knlCheck);
    for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        std::set<std::string> buildOption = mResource->buildOptions;
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[knl_idx], buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx].setArg(idx++, openCLImage(input));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(mResource->filter.get()));
        ret |= kernel[knl_idx].setArg(idx++, *mResource->dequantScaleBuffer.get());
        ret |= kernel[knl_idx].setArg(idx++, *mResource->dequantOffsetBuffer.get());
        ret |= kernel[knl_idx].setArg(idx++, *mResource->biasBuffer.get());
        ret |= kernel[knl_idx].setArg(idx++, openCLImage(output));
        ret |= kernel[knl_idx].setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= kernel[knl_idx].setArg(idx++, inputChannelBlocks);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(strideShape), strideShape);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(dilationShape), dilationShape);
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(width, itemW[knl_idx]));
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(outputShape.at(3), 4));
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(height, itemH[knl_idx]));
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvLowMemory Kernel Select");
        std::pair<std::vector<uint32_t>, int> retTune;
        retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
        if(min_cost.first > retTune.second) {
            min_cost.first = retTune.second;
            min_cost.second = knl_idx;
            mLocalWorkSize = {retTune.first[0], retTune.first[1]};
        }
    }
    int min_index  = min_cost.second;
    mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};

    std::set<std::string> buildOption = mResource->buildOptions;
    mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[min_index], buildOption);

    uint32_t idx            = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, openCLImage(input));
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->filter.get()));
    ret |= mKernel.setArg(idx++, *mResource->dequantScaleBuffer.get());
    ret |= mKernel.setArg(idx++, *mResource->dequantOffsetBuffer.get());
    ret |= mKernel.setArg(idx++, *mResource->biasBuffer.get());
    ret |= mKernel.setArg(idx++, openCLImage(output));
    ret |= mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
    ret |= mKernel.setArg(idx++, inputChannelBlocks);
    ret |= mKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= mKernel.setArg(idx++, sizeof(kernelShape), kernelShape);
    ret |= mKernel.setArg(idx++, sizeof(strideShape), strideShape);
    ret |= mKernel.setArg(idx++, sizeof(paddingShape), paddingShape);
    ret |= mKernel.setArg(idx++, sizeof(dilationShape), dilationShape);
    ret |= mKernel.setArg(idx++, UP_DIV(width, itemW[min_index]));
    ret |= mKernel.setArg(idx++, UP_DIV(outputShape.at(3), 4));
    ret |= mKernel.setArg(idx++, UP_DIV(height, itemH[min_index]));
    MNN_CHECK_CL_SUCCESS(ret, "setArg ConvLowMemory");
    mOpenCLBackend->recordKernel2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
    mOpenCLBackend->endRecord(mRecording);
    return;
}
void ConvLowMemoryExecution::tuneGemmLowMemory(Tensor * input, Tensor * output) {
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mOpenCLBackend->startRecord(mRecording);
    const int outChannel = outputShape.at(3);
    const int inputChannels = inputShape.at(3);
    const int batch = outputShape.at(0);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int outputChannelBlocks = UP_DIV(outChannel, 4);
    std::string kernelname = "gemm_conv";
    int global_x = outputChannelBlocks;
    int global_y = batch;
    if(batch > 1)
    {
        kernelname = "gemm_conv_b2";
        global_y = UP_DIV(batch, 2);
    }
    mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm", kernelname, mResource->buildOptions);
    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));
    mGlobalWorkSize = {static_cast<uint32_t>(global_x), static_cast<uint32_t>(global_y)};
    // MNN_PRINT("Kernel is %d.\n", min_index);
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, openCLImage(input));
    ret |= mKernel.setArg(idx++, *mResource->kernelBuffer.get());
    ret |= mKernel.setArg(idx++, *mResource->dequantScaleBuffer.get());
    ret |= mKernel.setArg(idx++, *mResource->dequantOffsetBuffer.get());
    ret |= mKernel.setArg(idx++, *mResource->biasBuffer.get());
    ret |= mKernel.setArg(idx++, openCLImage(output));
    ret |= mKernel.setArg(idx++, static_cast<int>(outputChannelBlocks));
    ret |= mKernel.setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= mKernel.setArg(idx++, static_cast<int>(batch));
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_conv");
    
    mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelname, mKernel).first;
    mOpenCLBackend->recordKernel2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
    mOpenCLBackend->endRecord(mRecording);
    return;
}
ConvLowMemoryExecution::ConvLowMemoryExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvLowMemoryExecution init !\n");
#endif
    mResource.reset(new ConvResource);
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dParams                  = conv2dParams;
    mResource->conv2dCommonParams  = conv2dCommonParams;
    mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    auto padding = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], conv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX

    mKernelWidth   = conv2dCommonParams->kernelX();
    mKernelHeight  = conv2dCommonParams->kernelY();
    mOutputChannel = conv2dCommonParams->outputCount();
    mInputChannel = inputs[0]->channel();
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    // set mDequantScale, mDequantOffset, mFilterDataPtr
    // prepare mDequantScale mDequantOffset mFilterDataPtr
    getInfoFromOpLowMemory(quanCommon);
    //select opt conv method
    //std::vector<int> inputShape  = tensorShapeFormat(inputs[0]);
    //const int inputChannels = inputShape.at(3);
    //const int batch = inputShape.at(0);
    mResource->gemmOpt = (mKernelHeight == mKernelWidth && mKernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0 && mStrides[0] == 1 && mStrides[1] == 1 && inputs[0]->width() == 1 && inputs[0]->height() == 1);
    mResource->conv1x1Opt = (mKernelHeight == mKernelWidth && mKernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0 && mStrides[0] == 1 && mStrides[1] == 1 && inputs[0]->width() >= 4);
        //printf("mConv1x1Opt = %d  mKernelHeight = %d  mKernelWidth = %d  mPaddings[0] = %d mPaddings[1] = %d mStrides[0] = %d mStrides[1] = %d inputs[0]->width() = %d inputs[0]->height() = %d mOutputChannel = %d inputChannels = %d batch = %d\n", mConv1x1Opt, mKernelHeight, mKernelWidth,
               //mPaddings[0], mPaddings[1], mStrides[0], mStrides[1], inputs[0]->width(), inputs[0]->height(), mOutputChannel, inputChannels, batch);
        if (mResource->conv1x1Opt) {
            // set mKernelBuffer for 1x1 case
            // At first, set packCout equal to 4
            set1x1WeightLowMemory(4, 4, mFilterDataPtr, quanCommon);
        } else if(mResource->gemmOpt){
            set1x1WeightLowMemory(4, 4, mFilterDataPtr, quanCommon);
        }else {
            // set mFilter for not 1x1 case
            setGeneralWeightLowMemory(mFilterDataPtr, quanCommon);
        }
    // Create Kernel
    mResource->buildOptions.emplace("-DBIAS");
    if (conv2dCommonParams->relu()) {
        mResource->buildOptions.emplace("-DRELU");
    } else if (conv2dCommonParams->relu6()) {
        mResource->buildOptions.emplace("-DRELU6");
    }
    if (mNumQuantBit == 8) {
        // int8 case
        mResource->buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT8");
    } else if (mNumQuantBit == 4){
        // int4 case
        mResource->buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT4");
    } else {/* More types to be supported. */}
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvLowMemoryExecution::ConvLowMemoryExecution(std::shared_ptr<ConvResource> resource, const Op* op, Backend *backend)
    : ConvCommonExecution(backend) {
    mResource = resource;
}

ConvLowMemoryExecution::~ConvLowMemoryExecution() {
    // Do nothing
}

bool ConvLowMemoryExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvLowMemoryExecution(mResource, op, bn);
    return true;
}

ErrorCode ConvLowMemoryExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];
    // auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->conv2dCommonParams);
    // mPaddings[0] = padding.second;//padY
    // mPaddings[1] = padding.first;//padX
    mPaddings[0] = 0;
    mPaddings[1] = 0;
    if (mResource->conv1x1Opt) {
        tune1x1CaseLowMemory(input, output);
    } else if(mResource->gemmOpt){
        tuneGemmLowMemory(input, output);
    } else {
        tuneGeneralCaseLowMemory(input, output);
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}
ErrorCode ConvLowMemoryExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onExecute !\n");
#endif
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime(), &event);
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Conv2D", event});
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        if(mOpenCLBackend->isDevideOpRecord())
            mOpenCLBackend->addRecord(mRecording);
#ifdef LOG_VERBOSE
        MNN_PRINT("End ConvExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    // gemm/gemv:
    // input : (batch, ic/4, 4)
    // weight: (ic/4, oc, 4)
    // output: (batch, oc, 4)
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_LOW_MEMORY */
