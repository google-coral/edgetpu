"""Model benchmark case list."""

MODEL_BENCHMARK_CASES = [
    {
        "benchmark_name": "BM_MobileNetV1",
        "tflite_cpu_filepath": "mobilenet_v1_1.0_224_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_MobileNetV1_25",
        "tflite_cpu_filepath": "mobilenet_v1_0.25_128_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_v1_0.25_128_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_MobileNetV1_50",
        "tflite_cpu_filepath": "mobilenet_v1_0.5_160_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_v1_0.5_160_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_MobileNetV1_75",
        "tflite_cpu_filepath": "mobilenet_v1_0.75_192_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_v1_0.75_192_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_MobileNetV1_L2Norm",
        "tflite_cpu_filepath": "mobilenet_v1_1.0_224_l2norm_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_MobileNetV2",
        "tflite_cpu_filepath": "mobilenet_v2_1.0_224_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_MobileNetV2INatPlant",
        "tflite_cpu_filepath": "mobilenet_v2_1.0_224_inat_plant_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_MobileNetV2INatInsect",
        "tflite_cpu_filepath": "mobilenet_v2_1.0_224_inat_insect_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_MobileNetV2INatBird",
        "tflite_cpu_filepath": "mobilenet_v2_1.0_224_inat_bird_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_SsdMobileNetV1",
        "tflite_cpu_filepath": "ssd_mobilenet_v1_coco_quant_postprocess.tflite",
        "tflite_edgetpu_filepath": "ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_SsdMobileNetV2",
        "tflite_cpu_filepath": "ssd_mobilenet_v2_coco_quant_postprocess.tflite",
        "tflite_edgetpu_filepath": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_FaceSsd",
        "tflite_cpu_filepath": "ssd_mobilenet_v2_face_quant_postprocess.tflite",
        "tflite_edgetpu_filepath": "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_InceptionV1",
        "tflite_cpu_filepath": "inception_v1_224_quant.tflite",
        "tflite_edgetpu_filepath": "inception_v1_224_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_InceptionV2",
        "tflite_cpu_filepath": "inception_v2_224_quant.tflite",
        "tflite_edgetpu_filepath": "inception_v2_224_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_InceptionV3",
        "tflite_cpu_filepath": "inception_v3_299_quant.tflite",
        "tflite_edgetpu_filepath": "inception_v3_299_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_InceptionV4",
        "tflite_cpu_filepath": "inception_v4_299_quant.tflite",
        "tflite_edgetpu_filepath": "inception_v4_299_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_EfficientNetEdgeTpuSmall",
        "tflite_cpu_filepath": "efficientnet-edgetpu-S_quant.tflite",
        "tflite_edgetpu_filepath": "efficientnet-edgetpu-S_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_EfficientNetEdgeTpuMedium",
        "tflite_cpu_filepath": "efficientnet-edgetpu-M_quant.tflite",
        "tflite_edgetpu_filepath": "efficientnet-edgetpu-M_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_EfficientNetEdgeTpuLarge",
        "tflite_cpu_filepath": "efficientnet-edgetpu-L_quant.tflite",
        "tflite_edgetpu_filepath": "efficientnet-edgetpu-L_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_Deeplab513Mv2Dm1_WithArgMax",
        "tflite_cpu_filepath": "deeplabv3_mnv2_pascal_quant.tflite",
        "tflite_edgetpu_filepath": "deeplabv3_mnv2_pascal_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_Deeplab513Mv2Dm05_WithArgMax",
        "tflite_cpu_filepath": "deeplabv3_mnv2_dm05_pascal_quant.tflite",
        "tflite_edgetpu_filepath": "deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_SsdMobileNetV1FineTunedPet",
        "tflite_cpu_filepath": "ssd_mobilenet_v1_fine_tuned_pet.tflite",
        "tflite_edgetpu_filepath": "ssd_mobilenet_v1_fine_tuned_pet_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_PostTrainingQuantizedSlimMobileNetEdgeTpu",
        "tflite_cpu_filepath": "mobilenet_edgetpu_1.0_224_post_training_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_edgetpu_1.0_224_post_training_quant_edgetpu.tflite",
    },
    {
        "benchmark_name": "BM_PostTrainingQuantizedSlimMobileNetEdgeTpuDm075",
        "tflite_cpu_filepath": "mobilenet_edgetpu_0.75_224_post_training_quant.tflite",
        "tflite_edgetpu_filepath": "mobilenet_edgetpu_0.75_224_post_training_quant_edgetpu.tflite",
    },
]
