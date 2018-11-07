#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "approximate_top_k_op.cc"

using namespace tensorflow;

REGISTER_OP("ApproximateTopK")
        .Input("all_embeddings: float32")
        .Input("target_embeddings: float32")
        .Output("neighbor_indices: int32")
        .Attr("k: int")
        .Attr("num_iters_per_update: int")
        .Attr("num_trees: int = 16")
        .Attr("metric: string = 'dot'")
        .Attr("seed: int = 0")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
                auto batch_size = c->Dim(c->input(1), 0);
                int k;
                TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
                c->set_output(0, c->MakeShape({batch_size, k}));
                return Status::OK();
        });

REGISTER_KERNEL_BUILDER(Name("ApproximateTopK").Device(DEVICE_CPU), ApproximateTopKOp);