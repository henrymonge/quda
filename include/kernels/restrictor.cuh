#include <color_spinor_field_order.h>
#include <block_reduce_helper.h>
#include <multigrid_helper.cuh>
#include <fast_intdiv.h>
#include <array.h>
#include <block_reduction_kernel.h>
#include <shared_memory_cache_helper.h>

namespace quda {

  using namespace quda::colorspinor;

  template <int fineColor, int coarseColor> constexpr int coarse_colors_per_thread()
  {
    // for fine grids (Nc=3) have more parallelism so can use more coarse strategy
    return fineColor != 3 ? 2 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;
    //coarseColor >= 8 && coarseColor % 8 == 0 ? 8 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;
  }

  constexpr int max_z_block() { return 12; }

  /**
     Geometry used by the row-mapped restrictor (RestrictorRow): the
     fine and coarse local extents, the aggregate size and the derived
     row and line counts.  ok is false when the row-mapped kernel
     cannot be used for this pair of fields.
  */
  struct RestrictRowGeometry {
    bool ok = false;
    int xcb = 0;                   // fine checkerboard sites per x-row (fine X / 2)
    int g = 0;                     // checkerboard sites per aggregate per x-row (aggregate X / 2)
    int x1 = 0, x2 = 0;            // fine Y and Z extents
    int xc0 = 0, xc1 = 0, xc2 = 0; // coarse X, Y and Z extents
    int bs1 = 0, bs2 = 0, bs3 = 0; // aggregate Y, Z and T extents
    int rows = 0;                  // x-rows per aggregate (bs1 * bs2 * bs3)
    int lines = 0;                 // lines of aggregates along x (coarse Y * Z * T)
  };

  /**
     @brief Compute the RestrictRowGeometry for restricting in to out.
     The row-mapped kernel requires a single-parity 4-d fine field, a
     full (both parity) coarse field, and an even aggregate extent in
     x, so that each aggregate owns a whole number of checkerboard
     sites in every x-row.
  */
  inline RestrictRowGeometry restrictRowGeometry(const ColorSpinorField &out, const ColorSpinorField &in)
  {
    RestrictRowGeometry geo;
    if (in.Ndim() != 4 || out.Ndim() != 4 || in.SiteSubset() != QUDA_PARITY_SITE_SUBSET
        || out.SiteSubset() != QUDA_FULL_SITE_SUBSET)
      return geo;

    int xf[4], xc[4], bs[4];
    for (int d = 0; d < 4; d++) {
      xf[d] = in.X()[d] * (d == 0 ? 2 : 1); // a single-parity field stores X / 2
      xc[d] = out.X()[d];
      if (xc[d] <= 0 || xf[d] % xc[d] != 0) return geo;
      bs[d] = xf[d] / xc[d];
    }
    if (bs[0] % 2 != 0) return geo;

    geo.xcb = xf[0] / 2;
    geo.g = bs[0] / 2;
    geo.x1 = xf[1];
    geo.x2 = xf[2];
    geo.xc0 = xc[0];
    geo.xc1 = xc[1];
    geo.xc2 = xc[2];
    geo.bs1 = bs[1];
    geo.bs2 = bs[2];
    geo.bs3 = bs[3];
    geo.rows = bs[1] * bs[2] * bs[3];
    geo.lines = xc[1] * xc[2] * xc[3];
    geo.ok = true;
    return geo;
  }

  /** 
      Kernel argument struct
  */
  template <typename out_t, typename in_t, typename v_t, int fineSpin_, int fineColor_, int coarseSpin_,
            int coarseColor_, bool from_non_rel_>
  struct RestrictArg : kernel_param<> {
    using real = out_t;
    static constexpr int fineSpin = fineSpin_;
    static constexpr int fineColor = fineColor_;
    static constexpr int coarseSpin = coarseSpin_;
    static constexpr int coarseColor = coarseColor_;
    static constexpr bool from_non_rel = from_non_rel_;

    // disable ghost to reduce arg size
    using F = FieldOrderCB<real, fineSpin, fineColor, 1, QUDA_NATIVE_FIELD_ORDER, in_t, in_t, true, isFixed<in_t>::value>;
    using C = FieldOrderCB<real, coarseSpin, coarseColor, 1, QUDA_NATIVE_FIELD_ORDER, out_t, out_t, true>;
    using V = FieldOrderCB<real, fineSpin, fineColor, coarseColor, QUDA_NATIVE_FIELD_ORDER, v_t>;

    const int_fastdiv n_src;
    C out[MAX_MULTI_RHS];
    F in[MAX_MULTI_RHS];
    const V v;
    const int aggregate_size;    // number of sites that form a single aggregate
    const int_fastdiv aggregate_size_cb; // number of checkerboard sites that form a single aggregate
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field

    // enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
    static constexpr bool swizzle = true;
    int_fastdiv swizzle_factor; // for transposing blockIdx.x mapping to coarse grid coordinate

    // geometry for the row-mapped kernel (RestrictorRow), unused by Restrictor
    int row_xcb = 0, row_g = 0, row_x1 = 0, row_x2 = 0;
    int row_xc0 = 0, row_xc1 = 0, row_xc2 = 0;
    int row_bs1 = 0, row_bs2 = 0, row_bs3 = 0, row_rows = 0;

    // when setting n_vector_z we can only use the compile-time part so can't include n_src in this min (not really an issue)
    static constexpr int n_vector_z = std::min(coarseColor/coarse_colors_per_thread<fineColor, coarseColor>(), max_z_block());
    static_assert(n_vector_z > 0, "n_vector_z cannot be less than 1");

    static constexpr bool launch_bounds = false;
    dim3 grid_dim;
    dim3 block_dim;

    RestrictArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, int parity) :
      kernel_param(dim3(in.Volume() / out.Volume(), 1,
                        out.size() * (coarseColor / coarse_colors_per_thread<fineColor, coarseColor>()))),
      n_src(out.size()),
      v(v),
      aggregate_size(in.Volume() / out.Volume()),
      aggregate_size_cb(in.VolumeCB() / out.Volume()),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      spin_map(),
      parity(parity),
      nParity(in.SiteSubset()),
      swizzle_factor(1)
    {
      if (out.size() > get_max_multi_rhs())
        errorQuda("vector set size %lu greater than max size %d", out.size(), get_max_multi_rhs());
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
      }

      auto geo = restrictRowGeometry(out[0], in[0]);
      row_xcb = geo.xcb;
      row_g = geo.g;
      row_x1 = geo.x1;
      row_x2 = geo.x2;
      row_xc0 = geo.xc0;
      row_xc1 = geo.xc1;
      row_xc2 = geo.xc2;
      row_bs1 = geo.bs1;
      row_bs2 = geo.bs2;
      row_bs3 = geo.bs3;
      row_rows = geo.rows;
    }
  };

  /**
     Rotates from the fine-color basis into the coarse-color basis.
  */
  template <typename Out, typename Arg>
  __device__ __host__ inline void rotateCoarseColor(Out &out, const Arg &arg, int src_idx, int parity, int x_cb, int coarse_color_block)
  {
    constexpr int coarse_color_per_thread = coarse_colors_per_thread<Arg::fineColor, Arg::coarseColor>();
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;
    const int v_parity = (arg.v.Nparity() == 2) ? parity : 0;

#pragma unroll
    for (int s = 0; s < Arg::fineSpin; s++)
#pragma unroll
      for (int coarse_color_local=0; coarse_color_local<coarse_color_per_thread; coarse_color_local++) {
	out[s*coarse_color_per_thread+coarse_color_local] = 0.0;
      }

#pragma unroll
    for (int coarse_color_local = 0; coarse_color_local < coarse_color_per_thread; coarse_color_local++) {
      int i = coarse_color_block + coarse_color_local;

      ColorSpinor<typename Arg::real, Arg::fineColor, Arg::fineSpin> in;
      arg.in[src_idx].template load<Arg::fineSpin>(in.data, spinor_parity, x_cb);

      if constexpr (Arg::fineSpin == 4 && Arg::from_non_rel) {
        in.toRel();
        in *= rsqrt(static_cast<typename Arg::real>(2.0));
      }

#pragma unroll
      for (int s = 0; s < Arg::fineSpin; s++) {
	for (int c = 0; c < Arg::fineColor; c++) {
          out[s * coarse_color_per_thread + coarse_color_local] =
            cmac(conj(arg.v(v_parity, x_cb, s, c, i)), in(s, c), out[s * coarse_color_per_thread + coarse_color_local]);
        }
      }
    }
  }

  template <typename Arg> struct RestrictorParams {
    static constexpr int coarse_color_per_thread = coarse_colors_per_thread<Arg::fineColor, Arg::coarseColor>();
    using vector = array<complex<typename Arg::real>, Arg::coarseSpin*coarse_color_per_thread>;
    static constexpr int block_dim = 1;
    using BlockReduce_t = BlockReduce<vector, block_dim, Arg::n_vector_z>;
  };
  template <typename Arg> struct Restrictor : KernelOps<typename RestrictorParams<Arg>::BlockReduce_t> {
    static constexpr int coarse_color_per_thread = RestrictorParams<Arg>::coarse_color_per_thread;
    using vector = typename RestrictorParams<Arg>::vector;
    using BlockReduce_t = typename RestrictorParams<Arg>::BlockReduce_t;
    const Arg &arg;
    using typename KernelOps<BlockReduce_t>::KernelOpsT;
    template <typename... Ops>
    constexpr Restrictor(const Arg &arg, const Ops &...ops) : KernelOpsT(ops...), arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    template <bool allthreads = false> // true if all threads in block will enter, even if out of range
    __device__ __host__ inline void operator()(dim3 block, dim3 thread, bool alive = true)
    {
      int x_fine_offset = thread.x;
      const int x_coarse = block.x;
      const int src_coarse_color_thread = block.z * arg.block_dim.z + thread.z;
      int src_idx = src_coarse_color_thread % arg.n_src;
      int coarse_color_thread = src_coarse_color_thread / arg.n_src;
      const int coarse_color_block = coarse_color_thread * coarse_color_per_thread;

      vector reduced{0};
      if (!allthreads || alive) {
        while (x_fine_offset < arg.aggregate_size) {
          // all threads with x_fine_offset greater than aggregate_size_cb are second parity
          const int parity_offset = x_fine_offset >= arg.aggregate_size_cb ? 1 : 0;
          const int x_fine_cb_offset = x_fine_offset % arg.aggregate_size_cb;
          const int parity = arg.nParity == 2 ? parity_offset : arg.parity;

          // look-up map is ordered as (coarse-block-id + fine-point-id),
          // with fine-point-id parity ordered
          const int x_fine_site_id = (x_coarse * 2 + parity) * arg.aggregate_size_cb + x_fine_cb_offset;
          const int x_fine = arg.coarse_to_fine[x_fine_site_id];
          const int x_fine_cb = x_fine - parity * arg.in[src_idx].VolumeCB();

          array<complex<typename Arg::real>, Arg::fineSpin * coarse_color_per_thread> tmp {0};

          rotateCoarseColor(tmp, arg, src_idx, parity, x_fine_cb, coarse_color_block);

          // perform any local spin coarsening
#pragma unroll
          for (int s = 0; s < Arg::fineSpin; s++) {
#pragma unroll
            for (int v = 0; v < coarse_color_per_thread; v++) {
              if (arg.spin_map(s, parity) == 0) {
                reduced[0 * coarse_color_per_thread + v] += tmp[s * coarse_color_per_thread + v];
              } else {
                reduced[1 * coarse_color_per_thread + v] += tmp[s * coarse_color_per_thread + v];
              }
            }
          }

          x_fine_offset += target::block_dim().x;
        }
      }

      reduced = BlockReduce_t(*this, thread.z).Sum(reduced);

      if (!allthreads || alive) {
        if (target::thread_idx().x == 0) {
          const int parity_coarse = static_cast<index_t>(x_coarse) >= arg.out[src_idx].VolumeCB() ? 1 : 0;
          const int x_coarse_cb = x_coarse - parity_coarse * arg.out[src_idx].VolumeCB();

#pragma unroll
          for (int s = 0; s < Arg::coarseSpin; s++) {
#pragma unroll
            for (int coarse_color_local = 0; coarse_color_local < coarse_color_per_thread; coarse_color_local++) {
              int v = coarse_color_thread * coarse_color_per_thread + coarse_color_local;
              arg.out[src_idx](parity_coarse, x_coarse_cb, s, v)
                = reduced[s * coarse_color_per_thread + coarse_color_local];
            }
          }
        }
      }
    }
  };

  /**
     Row-mapped restrictor.  Each thread block handles one line of
     aggregates along x (fixed coarse y, z, t).  Consecutive threads
     take consecutive fine checkerboard sites along an x-row, so the
     loads of the fine spinor and of V are contiguous, unlike
     Restrictor where the threads of a block gather the scattered
     sites of a single aggregate.  The block has block_dim.x / xcb row
     slices; each thread accumulates the aggregate its x position
     belongs to over the aggregate rows of its slice, and the partial
     sums are combined through shared memory.  Requires
     RestrictRowGeometry::ok.
  */
  template <typename Arg> struct RestrictorRow : KernelOps<SharedMemoryCache<typename RestrictorParams<Arg>::vector>> {
    static constexpr int coarse_color_per_thread = RestrictorParams<Arg>::coarse_color_per_thread;
    using vector = typename RestrictorParams<Arg>::vector;
    using Cache = SharedMemoryCache<vector>;
    const Arg &arg;
    using typename KernelOps<Cache>::KernelOpsT;
    template <typename... Ops>
    constexpr RestrictorRow(const Arg &arg, const Ops &...ops) : KernelOpsT(ops...), arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(dim3 block, dim3 thread)
    {
      const int src_coarse_color_thread = block.z * arg.block_dim.z + thread.z;
      const int src_idx = src_coarse_color_thread % arg.n_src;
      const int coarse_color_thread = src_coarse_color_thread / arg.n_src;
      const int coarse_color_block = coarse_color_thread * coarse_color_per_thread;
      const int parity = arg.parity;

      // line of aggregates handled by this block
      const int line = block.x;
      const int yc = line % arg.row_xc1;
      const int zc = (line / arg.row_xc1) % arg.row_xc2;
      const int tc = line / (arg.row_xc1 * arg.row_xc2);

      const int lane = thread.x % arg.row_xcb;           // checkerboard x position in the row
      const int slice = thread.x / arg.row_xcb;          // which subset of the aggregate rows
      const int n_slice = arg.block_dim.x / arg.row_xcb; // number of row slices in the block

      vector partial{0};
      for (int k = slice; k < arg.row_rows; k += n_slice) {
        const int y = yc * arg.row_bs1 + k % arg.row_bs1;
        const int z = zc * arg.row_bs2 + (k / arg.row_bs1) % arg.row_bs2;
        const int t = tc * arg.row_bs3 + k / (arg.row_bs1 * arg.row_bs2);
        const int x_fine_cb = ((t * arg.row_x2 + z) * arg.row_x1 + y) * arg.row_xcb + lane;

        array<complex<typename Arg::real>, Arg::fineSpin * coarse_color_per_thread> tmp{0};

        rotateCoarseColor(tmp, arg, src_idx, parity, x_fine_cb, coarse_color_block);

        // perform any local spin coarsening
#pragma unroll
        for (int s = 0; s < Arg::fineSpin; s++) {
#pragma unroll
          for (int v = 0; v < coarse_color_per_thread; v++) {
            if (arg.spin_map(s, parity) == 0) {
              partial[0 * coarse_color_per_thread + v] += tmp[s * coarse_color_per_thread + v];
            } else {
              partial[1 * coarse_color_per_thread + v] += tmp[s * coarse_color_per_thread + v];
            }
          }
        }
      }

      Cache cache {*this};
      cache.save(partial);
      cache.sync();

      // one thread per aggregate of the line sums its g lanes over all row slices
      if (static_cast<int>(thread.x) < arg.row_xc0) {
        const int a = thread.x;
        vector sum{0};
        for (int s = 0; s < n_slice; s++) {
          for (int j = 0; j < arg.row_g; j++) {
            const vector p = cache.load_x(s * arg.row_xcb + a * arg.row_g + j);
#pragma unroll
            for (int i = 0; i < Arg::coarseSpin * coarse_color_per_thread; i++) sum[i] += p[i];
          }
        }

        // coarse site of aggregate a, from the first fine site of its first row
        const int x_fine_cb0
          = (((tc * arg.row_bs3) * arg.row_x2 + zc * arg.row_bs2) * arg.row_x1 + yc * arg.row_bs1) * arg.row_xcb
          + a * arg.row_g;
        const int x_coarse = arg.fine_to_coarse[parity * arg.in[src_idx].VolumeCB() + x_fine_cb0];
        const int parity_coarse = x_coarse >= arg.out[src_idx].VolumeCB() ? 1 : 0;
        const int x_coarse_cb = x_coarse - parity_coarse * arg.out[src_idx].VolumeCB();

#pragma unroll
        for (int s = 0; s < Arg::coarseSpin; s++) {
#pragma unroll
          for (int coarse_color_local = 0; coarse_color_local < coarse_color_per_thread; coarse_color_local++) {
            int v = coarse_color_thread * coarse_color_per_thread + coarse_color_local;
            arg.out[src_idx](parity_coarse, x_coarse_cb, s, v) = sum[s * coarse_color_per_thread + coarse_color_local];
          }
        }
      }
    }
  };
}
