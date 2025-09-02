zig fetch --save https://github.com/attarrumi/ukfz/archive/refs/heads/main.zip

const matzig = b.dependency("ukfz", .{ .target = target, .optimize = optimize, });

exe.root_module.addImport("ukfz", matzig.module("ukfz"));
