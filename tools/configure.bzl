"""Rules for configuring the C++ cross-toolchain."""

def _impl(repository_ctx):
    gcc_version = repository_ctx.execute(["/bin/bash", "-c", "gcc -dumpversion | cut -f1 -d."]).stdout
    bcm2708_toolchain_root = repository_ctx.os.environ.get("BCM2708_TOOLCHAIN_ROOT", "/tools/arm-bcm2708/")
    repository_ctx.symlink(Label("//:BUILD.tpl"), "BUILD")
    repository_ctx.template(
        "cc_toolchain_config.bzl",
        Label("//:cc_toolchain_config.bzl.tpl"),
        {
            "%{gcc_version}": gcc_version,
            "%{bcm2708_toolchain_root}": bcm2708_toolchain_root,
        },
    )

cc_crosstool = repository_rule(
    environ = [
        "BCM2708_TOOLCHAIN_ROOT",
    ],
    implementation = _impl,
    local = True,
)
