#include "tensorflow/core/framework/op_kernel.h"
#include "annoylib.h"
#include "kissrandom.h"

using namespace tensorflow;

class ApproximateTopKOp : public OpKernel {
public:
    explicit ApproximateTopKOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_negative_samples", &num_negative_samples));
        OP_REQUIRES_OK(context, context->GetAttr("num_trees", &num_trees));
        OP_REQUIRES_OK(context, context->GetAttr("num_dims", &num_dims));
        OP_REQUIRES_OK(context, context->GetAttr("num_iters_per_update", &num_iters_per_update));
        OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
        OP_REQUIRES_OK(context, context->GetAttr("metric", &metric));
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& all_embeddings = context->input(0);
        auto all = all_embeddings.flat<float>();

        OP_REQUIRES(context, all_embeddings.dims() == expected_num_dims,
                    errors::InvalidArgument("Input all_embeddings tensor has to have 2 dimensions"));

        const Tensor& target_embeddings = context->input(1);
        auto target = target_embeddings.flat<float>();

        OP_REQUIRES(context, target_embeddings.dims() == expected_num_dims,
                    errors::InvalidArgument("Input target_embeddings tensor has to have 2 dimensions"));

        OP_REQUIRES(context, all_embeddings.dim_size(1) == target_embeddings.dim_size(1),
                    errors::InvalidArgument("Both inputs should have the same size of the second dimension"));

        const int64 num_items = all_embeddings.dim_size(0);
        auto vec = (double *) malloc( num_dims * sizeof(double) );

        if (iters_counter % num_iters_per_update == 0) {
            // Indexing
            if (metric == "dot") {
                t.reset(new AnnoyIndex<int, double, DotProduct, Kiss32Random>(num_dims));
            } else if (metric == "euclidean") {
                t.reset(new AnnoyIndex<int, double, Euclidean, Kiss32Random>(num_dims));
            } else if (metric == "cosine") {
                t.reset(new AnnoyIndex<int, double, Angular, Kiss32Random>(num_dims));
            } else {
                OP_REQUIRES(
                        context,
                        false,
                        errors::InvalidArgument( "metric has to be one of the following: 'dot', 'euclidean' or 'cosine'" ));
            }

            t->set_seed(seed);
            for(int i=0; i<num_items; ++i){
                for(int z=0; z<num_dims; ++z){
                    vec[z] = (double)all(i*num_dims+z);
                }

                t->add_item(i, vec);
            }

            t->build(num_trees);
            iters_counter = 0;
        }

        // Create an output tensor
        Tensor* neighbor_indices = nullptr;
        TensorShape shape = TensorShape();
        shape.AddDim(target_embeddings.dim_size(0));
        shape.AddDim(num_negative_samples);

        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &neighbor_indices));
        auto output_flat = neighbor_indices->flat<int32>();
        const int64 num_target_items = target_embeddings.dim_size(0);
        std::vector<int> toplist;

        for(int i=0; i<num_target_items; ++i){
            for(int z=0; z<num_dims; ++z){
                vec[z] = (double)target(i*num_dims+z);
            }

            t->get_nns_by_vector(vec, (size_t) num_negative_samples, (size_t) -1, &toplist, nullptr);

            for(int z=0; z<num_negative_samples; z++){
                output_flat(i*num_negative_samples+z) = toplist[z];
            }

            toplist.clear();
        }
        ++iters_counter;
    }

private:
    const int expected_num_dims = 2;
    int num_trees;
    int num_negative_samples;
    int num_dims;
    int num_iters_per_update;
    int iters_counter = 0;
    string metric;
    int seed;
    std::unique_ptr<AnnoyIndexInterface<int, double>> t;
};