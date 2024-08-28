import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model 


# Tạo ImageDataGenerator để tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2, 

    horizontal_flip=True,
    fill_mode='nearest' 

)

# Tải mô hình VGG16 đã được huấn luyện sẵn
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Thêm các lớp fully-connected
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions) 


# Freeze các lớp của base model
for layer in base_model.layers:
    layer.trainable = False

# Compile mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(
    train_datagen.flow_from_directory( 

        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    ),
    epochs=10
)