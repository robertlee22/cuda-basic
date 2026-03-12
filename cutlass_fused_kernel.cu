template <typename accum_t, typename scalar_t, typename output_t>
struct fused_bias_relu_epilogue {

    // Data members pass additional arguments to epilogue
    scalar_t const *Bias;
    accum_t threshold;

    /// Constructor callable on host and device initializes data members
    inline __device__ __host__
    fused_bias_relu_epilogue(
        scalar_t const *Bias,
        accum_t threshold
    ): Bias(Bias), threshold(threshold) { }

    /// Applies bias + ReLu operation
    inline __device__ __host__
    output_t operator()(
        accum_t accumulator,  /// element of matrix product result
        output_t c,           /// element of source accumulator matrix C
        size_t idx            /// index of c element; may be used to load
                              /// elements from other identically-
                              /// structured matrices
        ) const {

        // Compute the result by scaling the matrix product, adding bias, 
        // and adding the scaled accumulator element.

        accum_t result = output_t(
            alpha * scalar_t(accumulator) +
            Bias[i] +                         // load and add the bias
            beta * scalar_t(c)
        );

        // apply clamping function
        return max(threshold, result);
    }
};


// New: define type for custom epilogue functor
typedef fused_bias_relu_epilogue_t<float, float, float> 
    bias_relu_epilogue_t;

/// Computes GEMM fused with Bias and ReLu operation
__global__ void gemm_bias_relu(
    ...,                                    /// GEMM parameters not shown 
    bias_relu_epilogue_t bias_relu_op) {    /// bias_relu_op constructed 
                                            /// by caller

    // Define the block_task type.
    typedef block_task<
        block_task_policy_t,          // same policy as previous example
        float,
        float,
        matrix_transform_t::NonTranspose,
        4,
        matrix_transform_t::NonTranspose,
        4,
        bias_relu_epilogue_t,         // New: custom epilogue functor type
        4,
        true
    > block_task_t ;

    // Declare statically-allocated shared storage
    __shared__ block_task_t::scratch_storage_t smem;
    
    // Construct and run the task
    block_task_t(
        reinterpret_cast(&smem),
        &smem,
        A,
        B,
        C,
        bias_relu_op,                 // New: custom epilogue object
        M,
        N,
        K).run();
}