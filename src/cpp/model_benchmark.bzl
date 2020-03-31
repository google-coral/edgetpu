"""Generate model benchmark source file using template.
"""

_TEMPLATE = "//src/cpp:models_benchmark.cc.template"

def _generate_models_benchmark_src_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file._template,
        output = ctx.outputs.source_file,
        substitutions = {
            "{BENCHMARK_NAME}": ctx.attr.benchmark_name,
            "{TFLITE_CPU_FILEPATH}": ctx.attr.tflite_cpu_filepath,
            "{TFLITE_EDGETPU_FILEPATH}": ctx.attr.tflite_edgetpu_filepath,
        },
    )

generate_models_benchmark_src = rule(
    implementation = _generate_models_benchmark_src_impl,
    attrs = {
        "benchmark_name": attr.string(mandatory = True),
        "tflite_cpu_filepath": attr.string(mandatory = True),
        "tflite_edgetpu_filepath": attr.string(mandatory = True),
        "_template": attr.label(
            default = Label(_TEMPLATE),
            allow_single_file = True,
        ),
    },
    outputs = {"source_file": "%{name}.cc"},
)
