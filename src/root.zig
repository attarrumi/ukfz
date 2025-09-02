const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const matzig = @import("mzig");
// Matrix structure
pub const Matrix = matzig.Matrix;

const Error = error{
    AllocationFailed,
    SingularMatrix,
    InvalidDimension,
};

const FilterFn = *const fn (dt: f32, x: Matrix, allocator: std.mem.Allocator) anyerror!Matrix;

pub const UnscentedKalmanFilter = struct {
    dim_x: usize,
    dim_z: usize,
    dt: f32,
    fx: FilterFn,
    hx: FilterFn,
    x: Matrix,
    p: Matrix,
    r: Matrix,
    q: Matrix,
    prior_x: Matrix,
    prior_p: Matrix,
    wm: Matrix,
    wc: Matrix,
    sigma_xs: Matrix,
    sigma_params: [3]f32,
    s: Matrix,
    y: Matrix,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        dim_x: usize,
        dim_z: usize,
        dt: f32,
        fx: FilterFn,
        hx: FilterFn,
    ) !UnscentedKalmanFilter {
        const x = try Matrix.zeros(allocator, dim_x, 1);
        const p = try Matrix.eye(allocator, dim_x);
        const r = try Matrix.zeros(allocator, dim_z, dim_z);
        const q = try Matrix.zeros(allocator, dim_x, dim_x);
        const prior_x = try Matrix.zeros(allocator, dim_x, 1);
        const prior_p = try Matrix.eye(allocator, dim_x);
        const s = try Matrix.zeros(allocator, dim_z, dim_z);
        const y = try Matrix.zeros(allocator, dim_z, 1);
        const sigma_xs = try Matrix.zeros(allocator, dim_x, 2 * dim_x + 1);

        var filter = UnscentedKalmanFilter{
            .dim_x = dim_x,
            .dim_z = dim_z,
            .dt = dt,
            .fx = fx,
            .hx = hx,
            .x = x,
            .p = p,
            .r = r,
            .q = q,
            .prior_x = prior_x,
            .prior_p = prior_p,
            .wm = try Matrix.zeros(allocator, 1, 1),
            .wc = try Matrix.zeros(allocator, 1, 1),
            .sigma_xs = sigma_xs,
            .sigma_params = [3]f32{ 0.1, 0.0, 2.0 },
            .s = s,
            .y = y,
            .allocator = allocator,
        };

        const weights = try filter.computeSigmaWeights();
        filter.wm.deinit();
        filter.wc.deinit();
        filter.wm = weights[0];
        filter.wc = weights[1];

        return filter;
    }

    pub fn deinit(self: *UnscentedKalmanFilter) void {
        self.x.deinit();
        self.p.deinit();
        self.r.deinit();
        self.q.deinit();
        self.prior_x.deinit();
        self.prior_p.deinit();
        self.wm.deinit();
        self.wc.deinit();
        self.sigma_xs.deinit();
        self.s.deinit();
        self.y.deinit();
    }

    fn computeSigmaWeights(self: *UnscentedKalmanFilter) ![2]Matrix {
        const n = self.dim_x;
        const alpha = self.sigma_params[0];
        const kappa = self.sigma_params[1];
        const beta = self.sigma_params[2];

        const nf: f32 = @floatFromInt(n);
        const lambda = (alpha * alpha) * (nf + kappa) - nf;

        const m = 2 * n + 1;
        const c = 0.5 / (nf + lambda);

        var wm_arr = try self.allocator.alloc(f32, m);
        defer self.allocator.free(wm_arr); // Fixed: use defer for cleanup
        var wc_arr = try self.allocator.alloc(f32, m);
        defer self.allocator.free(wc_arr); // Fixed: use defer for cleanup

        for (0..m) |i| {
            wm_arr[i] = c;
            wc_arr[i] = c;
        }

        wm_arr[0] = lambda / (nf + lambda);
        wc_arr[0] = lambda / (nf + lambda) + (1.0 - alpha * alpha + beta);

        const wm = try Matrix.initWithData(self.allocator, 1, m, wm_arr);
        const wc = try Matrix.initWithData(self.allocator, 1, m, wc_arr);

        return [2]Matrix{ wm, wc };
    }

    fn computeSigmaPoints(self: *UnscentedKalmanFilter, x: *const Matrix, p: *const Matrix) !Matrix {
        const n = self.dim_x;
        const alpha = self.sigma_params[0];
        const kappa = self.sigma_params[1];

        const m = 2 * n + 1;
        var sigmas = try Matrix.init(self.allocator, n, m);

        sigmas.setColumn(0, x.*);

        const nf: f32 = @floatFromInt(n);
        const lambda = (alpha * alpha) * (nf + kappa) - nf;

        var L = try p.cholesky(self.allocator);
        defer L.deinit(); // Fixed: use defer
        const s = @sqrt(lambda + nf);
        var S = try L.scale(s, self.allocator);
        defer S.deinit(); // Fixed: use defer

        for (0..n) |k| {
            var sk = try S.getColumn(k, self.allocator);
            defer sk.deinit(); // Fixed: use defer
            var x_plus = try x.add(sk, self.allocator);
            defer x_plus.deinit(); // Fixed: use defer
            var x_minus = try x.sub(sk, self.allocator);
            defer x_minus.deinit(); // Fixed: use defer

            sigmas.setColumn(k + 1, x_minus);
            sigmas.setColumn(k + n + 1, x_plus);
        }

        return sigmas;
    }

    pub fn predict(self: *UnscentedKalmanFilter) !void {
        const n = self.dim_x;
        var sigmas = try self.computeSigmaPoints(&self.x, &self.p);
        defer sigmas.deinit(); // Fixed: cleanup sigmas

        self.sigma_xs.deinit();
        self.sigma_xs = try Matrix.init(self.allocator, n, sigmas.cols);

        const c = sigmas.cols;
        for (0..c) |j| {
            var sigma_col = try sigmas.getColumn(j, self.allocator);
            defer sigma_col.deinit(); // Fixed: use defer
            var y = try self.fx(self.dt, sigma_col, self.allocator);
            defer y.deinit(); // Fixed: use defer
            self.sigma_xs.setColumn(j, y);
        }

        // compute prior_x (mean)
        self.prior_x.deinit();
        self.prior_x = try Matrix.zeros(self.allocator, n, 1);
        for (0..c) |j| {
            const w = self.wm.get(0, j);
            for (0..n) |i| {
                const val = self.prior_x.get(i, 0) + self.sigma_xs.get(i, j) * w;
                self.prior_x.set(i, 0, val);
            }
        }

        // compute prior_p
        self.prior_p.deinit();
        self.prior_p = try self.q.copy(self.allocator);
        for (0..c) |j| {
            var sigma_col = try self.sigma_xs.getColumn(j, self.allocator);
            defer sigma_col.deinit(); // Fixed: use defer
            var diff_x = try sigma_col.sub(self.prior_x, self.allocator);
            defer diff_x.deinit(); // Fixed: use defer
            var diff_x_t = try diff_x.transpose(self.allocator);
            defer diff_x_t.deinit(); // Fixed: use defer
            var outer = try diff_x.mul(diff_x_t, self.allocator);
            defer outer.deinit(); // Fixed: use defer
            const w = self.wc.get(0, j);
            var scaled = try outer.scale(w, self.allocator);
            defer scaled.deinit(); // Fixed: use defer

            const new_prior_p = try self.prior_p.add(scaled, self.allocator);
            self.prior_p.deinit();
            self.prior_p = new_prior_p;
        }
    }

    pub fn update(self: *UnscentedKalmanFilter, z: *const Matrix) !Matrix {
        const n = self.dim_x;
        const m = self.dim_z;
        const count = self.sigma_xs.cols;

        // Transform sigma points through measurement
        var sigma_zs = try Matrix.init(self.allocator, m, count);
        defer sigma_zs.deinit();
        for (0..count) |j| {
            var sigma_col = try self.sigma_xs.getColumn(j, self.allocator);
            defer sigma_col.deinit(); // Fixed: use defer
            var z_pred = try self.hx(self.dt, sigma_col, self.allocator);
            defer z_pred.deinit(); // Fixed: use defer
            sigma_zs.setColumn(j, z_pred);
        }

        // predicted measurement mean prior_z
        var prior_z = try Matrix.zeros(self.allocator, m, 1);
        defer prior_z.deinit(); // Fixed: use defer
        for (0..count) |j| {
            const w = self.wm.get(0, j);
            var col = try sigma_zs.getColumn(j, self.allocator);
            defer col.deinit(); // Fixed: use defer
            var weighted = try col.scale(w, self.allocator);
            defer weighted.deinit(); // Fixed: use defer
            const new_prior = try prior_z.add(weighted, self.allocator);
            prior_z.deinit();
            prior_z = new_prior;
        }

        // covariances
        var pzz = try self.r.copy(self.allocator);
        defer pzz.deinit(); // Fixed: use defer
        var pxz = try Matrix.zeros(self.allocator, n, m);
        defer pxz.deinit(); // Fixed: use defer

        for (0..count) |j| {
            const w = self.wc.get(0, j);
            var z_col = try sigma_zs.getColumn(j, self.allocator);
            defer z_col.deinit(); // Fixed: use defer
            var diff_z = try z_col.sub(prior_z, self.allocator);
            defer diff_z.deinit(); // Fixed: use defer
            var diff_z_t = try diff_z.transpose(self.allocator);
            defer diff_z_t.deinit(); // Fixed: use defer
            var pzz_contrib = try diff_z.mul(diff_z_t, self.allocator);
            defer pzz_contrib.deinit(); // Fixed: use defer
            var scaled_pzz = try pzz_contrib.scale(w, self.allocator);
            defer scaled_pzz.deinit(); // Fixed: use defer

            const new_pzz = try pzz.add(scaled_pzz, self.allocator);
            pzz.deinit();
            pzz = new_pzz;

            // pxz
            var x_col = try self.sigma_xs.getColumn(j, self.allocator);
            defer x_col.deinit(); // Fixed: use defer
            var diff_x = try x_col.sub(self.prior_x, self.allocator);
            defer diff_x.deinit(); // Fixed: use defer
            var pxz_contrib = try diff_x.mul(diff_z_t, self.allocator);
            defer pxz_contrib.deinit(); // Fixed: use defer
            var scaled_pxz = try pxz_contrib.scale(w, self.allocator);
            defer scaled_pxz.deinit(); // Fixed: use defer

            const new_pxz = try pxz.add(scaled_pxz, self.allocator);
            pxz.deinit();
            pxz = new_pxz;
        }

        // invert pzz
        var pzz_inv = try pzz.inverse(self.allocator);
        defer pzz_inv.deinit(); // Fixed: use defer

        var K = try pxz.mul(pzz_inv, self.allocator);
        defer K.deinit(); // Fixed: use defer

        // innovation y = z - prior_z
        self.y.deinit();
        self.y = try z.sub(prior_z, self.allocator);

        var K_y = try K.mul(self.y, self.allocator);
        defer K_y.deinit(); // Fixed: use defer

        self.x.deinit();
        self.x = try self.prior_x.add(K_y, self.allocator);

        // P = prior_p - K * pzz * K^T
        var K_t = try K.transpose(self.allocator);
        defer K_t.deinit(); // Fixed: use defer
        var K_pzz = try K.mul(pzz, self.allocator);
        defer K_pzz.deinit(); // Fixed: use defer
        var K_pzz_Kt = try K_pzz.mul(K_t, self.allocator);
        defer K_pzz_Kt.deinit(); // Fixed: use defer

        self.p.deinit();
        self.p = try self.prior_p.sub(K_pzz_Kt, self.allocator);

        // s = pzz (store predicted measurement covariance)
        self.s.deinit();
        self.s = try pzz.copy(self.allocator);

        return self.x.copy(self.allocator);
    }
};

// Fixed the blockDiag function - it was missing defer
fn blockDiag(allocator: std.mem.Allocator, matrices: []const Matrix) !Matrix {
    var total_r: usize = 0;
    var total_c: usize = 0;
    for (matrices) |m| {
        total_r += m.rows;
        total_c += m.cols;
    }
    var res = try Matrix.init(allocator, total_r, total_c);
    var r_off: usize = 0;
    var c_off: usize = 0;
    for (matrices) |m| {
        for (0..m.rows) |r| {
            for (0..m.cols) |c| {
                res.set(r_off + r, c_off + c, m.get(r, c));
            }
        }
        r_off += m.rows;
        c_off += m.cols;
    }
    return res;
}

pub fn qDiscreteWhiteNoise(
    allocator: std.mem.Allocator,
    dim: usize,
    dt: f32,
    variance: f32,
) !Matrix {
    if (dim < 2 or dim > 4) return Error.InvalidDimension;
    const dt2 = dt * dt;
    const dt3 = dt2 * dt;
    const dt4 = dt3 * dt;

    var q_data: []f32 = undefined;
    switch (dim) {
        2 => {
            q_data = try allocator.alloc(f32, 4);
            defer allocator.free(q_data); // Fixed: use defer
            q_data[0] = 0.25 * dt4;
            q_data[1] = 0.5 * dt3;
            q_data[2] = 0.5 * dt3;
            q_data[3] = dt2;
        },
        3 => {
            q_data = try allocator.alloc(f32, 9);
            defer allocator.free(q_data); // Fixed: use defer
            q_data[0] = 0.25 * dt4;
            q_data[1] = 0.5 * dt3;
            q_data[2] = 0.5 * dt2;
            q_data[3] = 0.5 * dt3;
            q_data[4] = dt2;
            q_data[5] = dt;
            q_data[6] = 0.5 * dt2;
            q_data[7] = dt;
            q_data[8] = 1.0;
        },
        4 => {
            const dt5 = dt4 * dt;
            const dt6 = dt5 * dt;
            q_data = try allocator.alloc(f32, 16);
            defer allocator.free(q_data); // Fixed: use defer
            q_data[0] = dt6 / 36.0;
            q_data[1] = dt5 / 12.0;
            q_data[2] = dt4 / 6.0;
            q_data[3] = dt3 / 6.0;
            q_data[4] = dt5 / 12.0;
            q_data[5] = dt4 / 4.0;
            q_data[6] = dt3 / 2.0;
            q_data[7] = dt2 / 2.0;
            q_data[8] = dt4 / 6.0;
            q_data[9] = dt3 / 2.0;
            q_data[10] = dt2;
            q_data[11] = dt;
            q_data[12] = dt3 / 6.0;
            q_data[13] = dt2 / 2.0;
            q_data[14] = dt;
            q_data[15] = 1.0;
        },
        else => return Error.InvalidDimension,
    }

    var Q = try Matrix.initWithData(allocator, dim, dim, q_data);
    const scaled = try Q.scale(variance, allocator);
    Q.deinit(); // Fixed: cleanup Q before returning scaled
    return scaled;
}
