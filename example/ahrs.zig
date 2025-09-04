const std = @import("std");
const math = std.math;
const ukfz = @import("ukfz");

const Matrix = ukfz.Matrix;

// Quaternion utilities
pub const Quaternion = struct {
    w: f32,
    x: f32,
    y: f32,
    z: f32,

    const Self = @This();

    pub fn init(w: f32, x: f32, y: f32, z: f32) Self {
        return Self{ .w = w, .x = x, .y = y, .z = z };
    }

    pub fn normalize(self: Self) Self {
        const norm_val = math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z);
        if (norm_val == 0) return self;
        return Self{
            .w = self.w / norm_val,
            .x = self.x / norm_val,
            .y = self.y / norm_val,
            .z = self.z / norm_val,
        };
    }

    pub fn multiply(self: Self, other: Self) Self {
        return Self{
            .w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            .x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            .y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            .z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        };
    }

    pub fn toMatrix(self: Self, allocator: std.mem.Allocator) !Matrix {
        var result = try Matrix.zeros(allocator, 4, 1);
        result.set(0, 0, self.w);
        result.set(1, 0, self.x);
        result.set(2, 0, self.y);
        result.set(3, 0, self.z);
        return result;
    }

    pub fn fromMatrix(matrix: Matrix) Self {
        return Self{
            .w = matrix.get(0, 0),
            .x = matrix.get(1, 0),
            .y = matrix.get(2, 0),
            .z = matrix.get(3, 0),
        };
    }

    pub fn toEuler(self: Self) struct { roll: f32, pitch: f32, yaw: f32 } {
        const sinr_cosp = 2 * (self.w * self.x + self.y * self.z);
        const cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y);
        const roll = math.atan2(sinr_cosp, cosr_cosp);

        const sinp = 2 * (self.w * self.y - self.z * self.x);
        const pi: f32 = math.pi / 2.0;
        const pitch = if (@abs(sinp) >= 1) math.copysign(pi, sinp) else math.asin(sinp);

        const siny_cosp = 2 * (self.w * self.z + self.x * self.y);
        const cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z);
        const yaw = math.atan2(siny_cosp, cosy_cosp);

        return .{
            .roll = roll * (180.0 / math.pi),
            .pitch = pitch * (180.0 / math.pi),
            .yaw = yaw * (180.0 / math.pi),
        };
    }
};

// Quaternion integration from angular velocity
pub fn quaternionFromGyro(q: Quaternion, wx: f32, wy: f32, wz: f32, dt: f32) Quaternion {
    const half_dt = dt * 0.5;
    const dq = Quaternion.init(
        1.0,
        wx * half_dt,
        wy * half_dt,
        wz * half_dt,
    );
    return q.multiply(dq).normalize();
}

// Expected gravity vector from quaternion
pub fn expectedGravity(q: Quaternion) struct { x: f32, y: f32, z: f32 } {
    return .{
        .x = 2.0 * (q.x * q.z - q.w * q.y),
        .y = 2.0 * (q.w * q.x + q.y * q.z),
        .z = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
    };
}

// Expected magnetic field vector from quaternion (assuming magnetic north)
pub fn expectedMagnetic(q: Quaternion, mag_declination: f32) struct { x: f32, y: f32, z: f32 } {
    const cos_dec = math.cos(mag_declination);
    const sin_dec = math.sin(mag_declination);

    // Rotate magnetic north vector by quaternion
    const mx = (q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z) * cos_dec +
        2.0 * (q.x * q.y + q.w * q.z) * sin_dec;
    const my = 2.0 * (q.x * q.y - q.w * q.z) * cos_dec +
        (q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z) * sin_dec;
    const mz = 2.0 * (q.x * q.z + q.w * q.y) * cos_dec +
        2.0 * (q.y * q.z - q.w * q.x) * sin_dec;

    return .{ .x = mx, .y = my, .z = mz };
}

const prev_gyro = struct {
    gx: f32 = 0,
    gy: f32 = 0,
    gz: f32 = 0,
}{};

pub const quaternion_fx = struct {
    var prev_g: @TypeOf(prev_gyro) = undefined;

    pub fn fx(dt: f32, x: Matrix, alloc: std.mem.Allocator) !Matrix {
        var result = try x.copy(alloc);

        // Extract quaternion and biases
        const q = Quaternion.fromMatrix(x);
        const bias_gx = x.get(4, 0);
        const bias_gy = x.get(5, 0);
        const bias_gz = x.get(6, 0);

        // Corrected gyro measurements
        const wx = prev_g.gx - bias_gx;
        const wy = prev_g.gy - bias_gy;
        const wz = prev_g.gz - bias_gz;

        // Integrate quaternion
        const new_q = quaternionFromGyro(q, wx, wy, wz, dt);

        // Update state
        result.set(0, 0, new_q.w);
        result.set(1, 0, new_q.x);
        result.set(2, 0, new_q.y);
        result.set(3, 0, new_q.z);

        // Gyro biases evolve slowly (random walk)
        // They remain mostly constant in the prediction step

        return result;
    }

    pub fn setGyro(gx: f32, gy: f32, gz: f32) void {
        prev_g.gx = gx;
        prev_g.gy = gy;
        prev_g.gz = gz;
    }
};

pub const quaternion_hx = struct {
    pub fn hx(dt: f32, x: Matrix, alloc: std.mem.Allocator) !Matrix {
        _ = dt;
        var result = try Matrix.zeros(alloc, 6, 1);

        // Extract quaternion from state
        const q = Quaternion.fromMatrix(x);

        // Expected gravity vector (accelerometer measurements)
        const grav = expectedGravity(q);
        result.set(0, 0, grav.x);
        result.set(1, 0, grav.y);
        result.set(2, 0, grav.z);

        // Expected magnetic field vector (magnetometer measurements)
        const mag_declination = 0.0133808576; // Your magnetic declination
        const mag = expectedMagnetic(q, mag_declination);
        result.set(3, 0, mag.x);
        result.set(4, 0, mag.y);
        result.set(5, 0, mag.z);

        return result;
    }
};
