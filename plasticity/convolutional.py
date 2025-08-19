import jax

def conv_layer(input_channels, output_channels, kernel_size, key):
    k1, k2 = jax.random.split(key)
    weight = jax.random.normal(k1, (kernel_size, kernel_size, input_channels, output_channels))
    bias = jax.random.normal(k2, (output_channels))
    return weight, bias

@jax.jit
def feedforward_conv(params, a, padding="SAME"):
    w, b = params

    y = jax.lax.conv_general_dilated(
        lhs=a,
        rhs=w,
        window_strides=(1,1),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    return y + b
2
