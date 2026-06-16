import triton
import triton.language as tl


@triton.jit
def flash_attention_forward(

    # Input and output memory pointers
    query_matrix_pointer,
    key_matrix_pointer,
    value_matrix_pointer,
    output_matrix_pointer,

    # Memory strides for the query matrix
    query_row_stride,
    query_column_stride,

    # Memory strides for the key matrix
    key_row_stride,
    key_column_stride,

    # Memory strides for the value matrix
    value_row_stride,
    value_column_stride,

    # Memory strides for the output matrix
    output_row_stride,
    output_column_stride,

    # Problem dimensions
    sequence_length,
    attention_head_dimension,

    # Compile-time block sizes
    query_block_size: tl.constexpr,
    key_block_size: tl.constexpr,
):

    # Determine which block of query tokens this Triton program handles.

    query_block_index = tl.program_id(0)

    # Compute the token indices for this query block.

    query_token_indices = (
        query_block_index * query_block_size
        + tl.arange(0, query_block_size)
    )


    # Compute the feature dimension indices.

    feature_dimension_indices = tl.arange(
        0,
        attention_head_dimension,
    )


    # Load the current block of query vectors into fast on-chip memory.

    query_vectors = tl.load(
        query_matrix_pointer
        + query_token_indices[:, None] * query_row_stride
        + feature_dimension_indices[None, :] * query_column_stride,
        mask=query_token_indices[:, None] < sequence_length,
        other=0.0,
    )

    # Initialize the running statistics required for the online softmax.

    running_softmax_maximum = tl.full(
        (query_block_size,),
        -float("inf"),
        tl.float32,
    )

    running_softmax_normalization = tl.zeros(
        (query_block_size,),
        tl.float32,
    )

    accumulated_output_vectors = tl.zeros(
        (query_block_size, attention_head_dimension),
        tl.float32,
    )

    # Iterate over the key/value matrices one block at a time.

    for key_block_start_index in range(
        0,
        sequence_length,
        key_block_size,
    ):

        # Compute token indices for the current key/value block.

        key_token_indices = (
            key_block_start_index
            + tl.arange(0, key_block_size)
        )

        # Load a block of key vectors.

        key_vectors = tl.load(
            key_matrix_pointer
            + key_token_indices[:, None] * key_row_stride
            + feature_dimension_indices[None, :] * key_column_stride,
            mask=key_token_indices[:, None] < sequence_length,
            other=0.0,
        )

        # --------------------------------------------------------------
        # Compute attention scores for only this block.
        #
        # This computes:
        #
        #     Query × Keyᵀ
        #
        # rather than constructing the full T × T matrix.
        # --------------------------------------------------------------

        attention_scores = tl.dot(
            query_vectors,
            tl.trans(key_vectors),
        )

        # Online softmax update.

        maximum_attention_score_in_current_block = tl.max(
            attention_scores,
            axis=1,
        )

        updated_running_softmax_maximum = tl.maximum(
            running_softmax_maximum,
            maximum_attention_score_in_current_block,
        )

        exponentiated_attention_scores = tl.exp(
            attention_scores
            - updated_running_softmax_maximum[:, None]
        )

        previous_softmax_rescaling_factor = tl.exp(
            running_softmax_maximum
            - updated_running_softmax_maximum
        )

        normalization_from_current_block = tl.sum(
            exponentiated_attention_scores,
            axis=1,
        )

        updated_running_softmax_normalization = (
            previous_softmax_rescaling_factor
            * running_softmax_normalization
            + normalization_from_current_block
        )

        # Load the corresponding value vectors.

        value_vectors = tl.load(
            value_matrix_pointer
            + key_token_indices[:, None] * value_row_stride
            + feature_dimension_indices[None, :] * value_column_stride,
            mask=key_token_indices[:, None] < sequence_length,
            other=0.0,
        )

        # Update the accumulated output.

        accumulated_output_vectors = (
            accumulated_output_vectors
            * (
                previous_softmax_rescaling_factor
                / updated_running_softmax_normalization
            )[:, None]
        )

        accumulated_output_vectors += (
            tl.dot(
                exponentiated_attention_scores,
                value_vectors,
            )
            / updated_running_softmax_normalization[:, None]
        )

        # Store the updated running statistics for the next block.

        running_softmax_maximum = (
            updated_running_softmax_maximum
        )

        running_softmax_normalization = (
            updated_running_softmax_normalization
        )

    # Store the final output vectors.

    tl.store(
        output_matrix_pointer
        + query_token_indices[:, None] * output_row_stride
        + feature_dimension_indices[None, :] * output_column_stride,
        accumulated_output_vectors,
        mask=query_token_indices[:, None] < sequence_length,
    )