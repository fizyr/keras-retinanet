import keras
import keras_resnet.models

import keras_retinanet.layers

import math

def classification_subnet(num_classes=21, num_anchors=9, feature_size=256, prob_pi=0.01):
	options = {
		'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
		'bias_initializer': keras.initializers.zeros()
	}

	layers = []
	for i in range(4):
		layers.append(keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', activation='relu', name='cls_{}'.format(i), **options))
	layers.append(keras.layers.Conv2D(
		filters=num_classes * num_anchors,
		kernel_size=(3, 3),
		strides=1,
		padding='same',
		name='pyramid_classification',
		kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
		bias_initializer=keras.initializers.constant(-math.log((1 - prob_pi) / prob_pi)),
	))

	return layers

def regression_subnet(num_anchors=9, feature_size=256):
	options = {
		'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
		'bias_initializer': keras.initializers.zeros()
	}

	layers = []
	for i in range(4):
		layers.append(keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', activation='relu', name='reg_{}'.format(i), **options))
	layers.append(keras.layers.Conv2D(num_anchors * 4, (3, 3), strides=1, padding='same', name='pyramid_regression'))

	return layers

def compute_pyramid_features(res3, res4, res5, feature_size=256):
	# compute deconvolution kernel size based on scale
	scale = 2
	kernel_size = (2 * scale - scale % 2)

	# upsample res5 to get P5 from the FPN paper
	P5 = keras.layers.Conv2D(feature_size, (1, 1), strides=1, padding='same', name='P5')(res5)

	# upsample P5 and add elementwise to C4
	P5_upsampled = keras.layers.Conv2DTranspose(feature_size, kernel_size=kernel_size, strides=scale, padding='same', name='P5_upsampled')(P5)
	P4 = keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', name='res4_reduced')(res4)
	P4 = keras.layers.Add(name='P4')([P5_upsampled, P4])

	# upsample P4 and add elementwise to C3
	P4_upsampled = keras.layers.Conv2DTranspose(feature_size, kernel_size=kernel_size, strides=scale, padding='same', name='P4_upsampled')(P4)
	P3 = keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', name='res3_reduced')(res3)
	P3 = keras.layers.Add(name='P3')([P4_upsampled, P3])

	# "P6 is obtained via a 3x3 stride-2 conv on C5"
	P6 = keras.layers.Conv2D(feature_size, (3, 3), strides=2, padding='same', name='P6')(res5)

	# "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
	P7 = keras.layers.Activation('relu', name='res6_relu')(P6)
	P7 = keras.layers.Conv2D(feature_size, (3, 3), strides=2, padding='same', name='P7')(P7)

	return P3, P4, P5, P6, P7

def RetinaNet(inputs, backbone, num_classes=21, feature_size=256, *args, **kwargs):
	image, im_info, gt_boxes = inputs

	# TODO: Parametrize this
	num_anchors = 9
	_, res3, res4, res5 = backbone.outputs # we ignore res2

	# compute pyramid features as per https://arxiv.org/abs/1708.02002
	pyramid_features = compute_pyramid_features(res3, res4, res5)
	strides = [8, 16, 32, 64, 128]

	# construct classification and regression subnets
	classification_layers = classification_subnet(num_classes=num_classes, num_anchors=num_anchors, feature_size=feature_size)
	regression_layers     = regression_subnet(num_anchors=num_anchors, feature_size=feature_size)

	# for all pyramid levels, run classification and regression branch and compute anchors
	classification    = None
	labels            = None
	regression        = None
	regression_target = None
	anchors           = None
	for i, (p, s) in enumerate(zip(pyramid_features, strides)):
		# run the classification subnet
		cls = p
		for l in classification_layers:
			cls = l(cls)

		# compute labels and bbox_reg_targets
		lb, r, a          = keras_retinanet.layers.AnchorTarget(stride=s, name='boxes_{}'.format(i))([cls, im_info, gt_boxes])
		anchors           = a if anchors == None else keras.layers.Concatenate(axis=0)([anchors, a])
		labels            = lb if labels == None else keras.layers.Concatenate(axis=0)([labels, lb])
		regression_target = r if regression_target == None else keras.layers.Concatenate(axis=0)([regression_target, r])

		cls            = keras.layers.Reshape((-1, num_classes), name='classification_{}'.format(i))(cls)
		classification = cls if classification == None else keras.layers.Concatenate(axis=1)([classification, cls])

		# run the regression subnet
		#reg = p
		#for l in regression_layers:
		#	reg = l(reg)

		#reg        = keras.layers.Reshape((-1, 4), name='boxes_reshaped_{}'.format(i))(reg)
		#regression = reg if regression == None else keras.layers.Concatenate(axis=1)([regression, reg])

	# compute classification and regression losses
	classification = keras.layers.Activation('softmax', name='classification_softmax')(classification)
	cls_loss = keras_retinanet.layers.FocalLoss()([classification, labels, regression_target, regression_target])

	return keras.models.Model(inputs=inputs, outputs=[classification, labels, regression_target, cls_loss, anchors], *args, **kwargs)

def ResNet50RetinaNet(inputs, *args, **kwargs):
	image, _, _ = inputs
	resnet = keras_resnet.models.ResNet50(image, include_top=False)
	return RetinaNet(inputs, resnet, *args, **kwargs)
