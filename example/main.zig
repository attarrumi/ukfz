const std = @import("std");
const print = std.debug.print;
const math = std.math;
const ukfz = @import("ukfz");
const Matrix = ukfz.Matrix;
const fs = std.fs;

// State transition function untuk gelombang
fn waveStateTransition(dt: f32, x: Matrix, allocator: std.mem.Allocator) !Matrix {
    var x_new = try x.copy(allocator);

    const A = x.get(0, 0); // amplitude
    const f = x.get(1, 0); // frequency
    const phi = x.get(2, 0); // phase
    const A_dot = x.get(3, 0); // amplitude rate

    const omega = 2.0 * math.pi * f; // angular frequency

    // Update state
    x_new.set(0, 0, A + A_dot * dt); // A(k+1) = A(k) + Ȧ*dt
    x_new.set(1, 0, f); // f(k+1) = f(k) (constant frequency)
    x_new.set(2, 0, phi + omega * dt); // φ(k+1) = φ(k) + ω*dt
    x_new.set(3, 0, A_dot); // Ȧ(k+1) = Ȧ(k) (constant rate)

    return x_new;
}

// Measurement function - observasi amplitudo gelombang
fn waveMeasurement(dt: f32, x: Matrix, allocator: std.mem.Allocator) !Matrix {
    _ = dt; // unused
    var z = try Matrix.init(allocator, 1, 1);

    const A = x.get(0, 0); // amplitude
    // const f = x.get(1, 0); // frequency
    const phi = x.get(2, 0); // phase

    // const omega = 2.0 * math.pi * f;

    // Measured wave value: y = A * sin(φ)
    const wave_value = A * @sin(phi);
    z.set(0, 0, wave_value);

    return z;
}

// Fungsi untuk generate data gelombang sintetik dengan noise
fn generateWaveData(allocator: std.mem.Allocator, steps: usize, dt: f32) !struct { measurements: []f32, true_states: []f32 } {
    var measurements = try allocator.alloc(f32, steps);
    var true_states = try allocator.alloc(f32, steps * 4); // 4 states per step

    // Parameter gelombang sebenarnya
    var true_A: f32 = 2.0; // amplitude awal
    const true_f: f32 = 0.5; // frequency (Hz)
    var true_phi: f32 = 0.0; // phase awal
    const true_A_dot: f32 = 0.1; // amplitude growth rate

    const omega = 2.0 * math.pi * true_f;

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    for (0..steps) |i| {
        const t = @as(f32, @floatFromInt(i)) * dt;

        // Update true state
        true_A = 2.0 + true_A_dot * t; // growing amplitude
        true_phi = omega * t; // phase evolution

        // True wave value
        const true_wave = true_A * @sin(true_phi);

        // Add measurement noise
        const noise = random.floatNorm(f32) * 0.2; // Gaussian noise std=0.2
        measurements[i] = true_wave + noise;

        // Store true states
        true_states[i * 4 + 0] = true_A;
        true_states[i * 4 + 1] = true_f;
        true_states[i * 4 + 2] = true_phi;
        true_states[i * 4 + 3] = true_A_dot;
    }

    return .{ .measurements = measurements, .true_states = true_states };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Wave Tracking dengan Unscented Kalman Filter ===\n\n", .{});

    // Inside main(), after initializations:
    const file = try std.fs.cwd().createFile(
        "wave_data.csv",
        .{ .read = true },
    );
    defer file.close();
    // Write CSV header
    try file.writeAll("time,true_A,est_A,true_f,est_f,true_phi,est_phi,true_wave,measured,est_wave\n");

    // Parameter simulasi
    const dt: f32 = 0.1; // time step (seconds)
    const steps: usize = 100; // simulation steps
    const dim_x: usize = 4; // state dimension [A, f, φ, Ȧ]
    const dim_z: usize = 1; // measurement dimension [wave_value]

    // Generate synthetic wave data
    const data = try generateWaveData(allocator, steps, dt);
    defer allocator.free(data.measurements);
    defer allocator.free(data.true_states);

    // Initialize UKF
    var ukf = try ukfz.UnscentedKalmanFilter.init(allocator, dim_x, dim_z, dt, waveStateTransition, waveMeasurement);
    defer ukf.deinit();

    // Set initial state estimate [A, f, φ, Ȧ]
    ukf.x.set(0, 0, 1.5); // initial amplitude guess
    ukf.x.set(1, 0, 0.6); // initial frequency guess
    ukf.x.set(2, 0, 0.1); // initial phase guess
    ukf.x.set(3, 0, 0.05); // initial amplitude rate guess

    // Set initial covariance
    ukf.p.set(0, 0, 1.0); // amplitude uncertainty
    ukf.p.set(1, 1, 0.1); // frequency uncertainty
    ukf.p.set(2, 2, 1.0); // phase uncertainty
    ukf.p.set(3, 3, 0.5); // amplitude rate uncertainty

    // Process noise Q - model uncertainty
    ukf.q.set(0, 0, 0.01); // amplitude process noise
    ukf.q.set(1, 1, 0.001); // frequency process noise
    ukf.q.set(2, 2, 0.05); // phase process noise
    ukf.q.set(3, 3, 0.01); // amplitude rate process noise

    // Measurement noise R
    ukf.r.set(0, 0, 0.04); // measurement noise variance (0.2^2)

    print("Time\tTrue_A\tEst_A\tTrue_f\tEst_f\tTrue_Wave\tMeasured\tError\n", .{});
    print("----\t------\t-----\t------\t-----\t---------\t--------\t-----\n", .{});

    var total_error: f32 = 0.0;

    // Main tracking loop
    for (0..steps) |i| {
        const t = @as(f32, @floatFromInt(i)) * dt;

        // Predict step
        try ukf.predict();

        // Update step with measurement
        var z_meas = try Matrix.init(allocator, 1, 1);
        defer z_meas.deinit();
        z_meas.set(0, 0, data.measurements[i]);

        var estimated_state = try ukf.update(&z_meas);
        defer estimated_state.deinit();

        // Extract true and estimated values
        const true_A = data.true_states[i * 4 + 0];
        const true_f = data.true_states[i * 4 + 1];
        const true_phi = data.true_states[i * 4 + 2];
        const true_wave = true_A * @sin(true_phi);

        const est_A = ukf.x.get(0, 0);
        const est_f = ukf.x.get(1, 0);
        const est_phi = ukf.x.get(2, 0);
        const est_wave = est_A * @sin(est_phi);

        const errors = @abs(true_wave - est_wave);
        total_error += errors;

        // Write results to CSV
        const datas = std.fmt.allocPrint(allocator, "{d:.3},{d:.3},{d:.3},{d:.3},{d:.3},{d:.3},{d:.3},{d:.3},{d:.3},{d:.3}\n", .{ t, true_A, est_A, true_f, est_f, true_phi, est_phi, true_wave, data.measurements[i], errors }) catch unreachable;
        defer allocator.free(datas);
        file.writeAll(datas) catch unreachable;

        if (i % 10 == 0 or i < 10) {
            print("{d:.1}\t{d:.3}\t{d:.3}\t{d:.3}\t{d:.3}\t{d:.3}\t\t{d:.3}\t\t{d:.3}\n", .{ t, true_A, est_A, true_f, est_f, true_wave, data.measurements[i], errors });
        }
    }

    const avg_error = total_error / @as(f32, @floatFromInt(steps));
    print("\n=== Hasil Akhir ===\n", .{});
    print("Average tracking error: {d:.4}\n", .{avg_error});

    // Final state comparison
    const final_idx = (steps - 1) * 4;
    print("Final True State: A={d:.3}, f={d:.3}, φ={d:.3}, Ȧ={d:.3}\n", .{ data.true_states[final_idx], data.true_states[final_idx + 1], data.true_states[final_idx + 2], data.true_states[final_idx + 3] });
    print("Final Estimated:  A={d:.3}, f={d:.3}, φ={d:.3}, Ȧ={d:.3}\n", .{ ukf.x.get(0, 0), ukf.x.get(1, 0), ukf.x.get(2, 0), ukf.x.get(3, 0) });

    // Uncertainty analysis
    print("\nFinal State Uncertainties (diagonal of P):\n", .{});
    print("σ_A={d:.3}, σ_f={d:.3}, σ_φ={d:.3}, σ_Ȧ={d:.3}\n", .{ @sqrt(ukf.p.get(0, 0)), @sqrt(ukf.p.get(1, 1)), @sqrt(ukf.p.get(2, 2)), @sqrt(ukf.p.get(3, 3)) });
}

// Test function untuk validasi
pub fn testWaveTracking() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Test Wave Function ===\n");

    // Test state transition
    var x = try Matrix.init(allocator, 4, 1);
    defer x.deinit();
    x.set(0, 0, 2.0); // A
    x.set(1, 0, 0.5); // f
    x.set(2, 0, 0.0); // φ
    x.set(3, 0, 0.1); // Ȧ

    var x_next = try waveStateTransition(0.1, x, allocator);
    defer x_next.deinit();

    print("Original state: A={d:.3}, f={d:.3}, φ={d:.3}, Ȧ={d:.3}\n", .{ x.get(0, 0), x.get(1, 0), x.get(2, 0), x.get(3, 0) });
    print("Next state:     A={d:.3}, f={d:.3}, φ={d:.3}, Ȧ={d:.3}\n", .{ x_next.get(0, 0), x_next.get(1, 0), x_next.get(2, 0), x_next.get(3, 0) });

    // Test measurement
    var z = try waveMeasurement(0.1, x, allocator);
    defer z.deinit();
    print("Measurement: {d:.3}\n", .{z.get(0, 0)});
}
