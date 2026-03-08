import argument
import numpy as np
import tensorflow as tf
from conditionalDecorator import conditional_decorator
from keras.backend import set_floatx

set_floatx("float32")

# Read command line arguments
args = argument.initArg()

if args.optimized:
    optimized = True
else:
    optimized = False


@conditional_decorator(tf.function, optimized)
def uocp_a_fun_x(x):
    return tf.math.polyval(
        [
            np.float32(1878.6244900261463),
            np.float32(-4981.580023016213),
            np.float32(516.2941996957871),
            np.float32(6452.38177755237),
            np.float32(-436.0524457974526),
            np.float32(1264.0514576769442),
            np.float32(-20918.656956191975),
            np.float32(12954.334261316431),
            np.float32(28871.72866007402),
            np.float32(-37943.83286204571),
            np.float32(34.11141793217983),
            np.float32(29363.16490602074),
            np.float32(-25774.496334571464),
            np.float32(11073.868226559767),
            np.float32(-2702.638445370805),
            np.float32(375.62895901410747),
            np.float32(-28.064663950113868),
            np.float32(1.1265244540945243),
        ],
        x,
    )


@conditional_decorator(tf.function, optimized)
def uocp_c_fun_x(x):
    return tf.math.polyval(
        [
            np.float32(-43309.69063512314),
            np.float32(122888.63938515769),
            np.float32(-69735.99554716503),
            np.float32(-59749.183217994185),
            np.float32(25744.002733171154),
            np.float32(15730.398058573825),
            np.float32(54021.915506318735),
            np.float32(-44566.03206954511),
            np.float32(64.32177924593454),
            np.float32(-7780.173422833786),
            np.float32(1117.4042221859695),
            np.float32(7387.492376558274),
            np.float32(-7237.289515884936),
            np.float32(-705.4465901574707),
            np.float32(17170.20236584321),
            np.float32(-42.60228181558803),
            np.float32(-23266.56994359366),
            np.float32(10810.92851132453),
            np.float32(2545.4065429021307),
            np.float32(1.6554268823619098),
            np.float32(751.3515882152476),
            np.float32(-4447.12851190078),
            np.float32(3727.268889820381),
            np.float32(-1331.1791971457515),
            np.float32(227.4712483170547),
            np.float32(-17.646894926746256),
            np.float32(0.8568207255402533),
            np.float32(-2.34505930698951),
            np.float32(5.059010555584711),
        ],
        x,
    )
