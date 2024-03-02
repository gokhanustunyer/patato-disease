from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Conv2DTranspose
from keras.models import load_model, model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.preprocessing.image import DirectoryIterator
from keras.preprocessing import image
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class DLModel:
    # Egitim sonrasi modelin kaydedilecegi klasor yolu
    TrainedModelPath = './trained_model'
    TrainedModelName = 'cnn_model'
    # Verilerin oldugu klasor yolu
    DataDirectory = './data/patato'

    def __init__(self) -> None:
        self.model = Sequential()
        self.layer_count = 0
        self.img_w, self.img_h = 40, 40
        self.class_count = 0

    def initialize_model(self, img_w, img_h):
        self.img_h = img_h
        self.img_w = img_w
        self.train_generator, self.test_generator, self.validation_generator = self.to_image_dataset('train', 'test', 'valid')
        self.class_count = len(self.train_generator.class_indices.values())

    def compile_and_fit(self, optimizer: str = 'Adam', loss: str = 'categorical_crossentropy', metrics: list = ['accuracy'], epochs: int = 25, save_model: bool = False) -> None:
        # Olusturulan modelin derlenmesi
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # Compile edilen modelin train datalariyla fit edilmesi
        self.model.fit(self.train_generator, epochs=epochs, validation_data=self.validation_generator)
        # Eger isteniyorsa modeli kaydet
        if save_model:
            DLModel.save_model(self.model)

    def fit(self, generator: DirectoryIterator, epochs: int):
        self.model.fit(generator, epochs=epochs)

    def add_conv_layer(self, activation: str, dropout: float = None, input_shape: tuple = None):
        # Katman icin filtre sayisinin hesaplanmasi
        filter_count = 2 ** (5+self.layer_count)

        # Ilk katman icin input_shape'in guncellenmesi
        if self.layer_count == 0:
            input_shape=(self.img_w, self.img_h, 3)

        # Katmanin eklenmesi
        if input_shape != None: 
            self.model.add(Conv2D(filter_count, (3, 3), activation=activation, input_shape = input_shape))
        else:
            self.model.add(Conv2D(filter_count, (3, 3), activation=activation))

        # MaxPooling'in eklenmesi
        self.model.add(MaxPooling2D((2, 2)))

        # Varsa dropout degerinin eklenmesi
        if dropout != None:
            self.model.add(Dropout(dropout))
        
        # Layer sayisinin guncellenmesi
        self.layer_count += 1

    def add_flatten_layer(self, activation1: str = 'relu', activation2: str = 'softmax'):
        self.model.add(Flatten())
        self.model.add(Dense(128, activation=activation1))
        self.model.add(Dense(self.class_count, activation=activation2))

    def to_image_dataset(self, train_path, test_path, valid_path):
        # Resmin islenecegi boyutlar
        img_height = 40
        img_width = 40

        # Egitim, validasyon ve test setlerinin dosya yollari
        train_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DLModel.DataDirectory, train_path)
        validation_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DLModel.DataDirectory, train_path)
        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DLModel.DataDirectory, test_path)

        # Eğitim, doğrulama ve test setleri için resim ön işleme ve artırma
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Verilerin resimlerden gerekli formatlar halinde alinmasi
        train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=32, class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height), batch_size=32, class_mode='categorical')
        test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(img_width, img_height), batch_size=32, class_mode='categorical')
        
        return train_generator, test_generator, validation_generator


    def generage_and_fit(self, path: str, target_size: tuple, batch_size: int = 32, class_mode: str = 'categorical'):
        car_gen = DLModel.generate_image_data(os.path.join(DLModel.DataDirectory, path), target_size, batch_size, class_mode)
        self.fit(car_gen, 5)

    def evaluate(self, print_acc: bool = False) -> tuple:
        score = self.model.evaluate(self.test_generator)
        if print_acc:
            print("Test Loss:", score[0])
            print("Test Accuracy:", score[1])
        return score[0], score[1]

    def train_w_tranformer(self):
        train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DLModel.DataDirectory, 'train')
        val_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DLModel.DataDirectory, 'train')

        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Görüntüleri yükleme ve boyutlandırma
        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(128, 128),
                batch_size=20,
                class_mode='input')

        val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(128, 128),
                batch_size=20,
                class_mode='input')
        
        # Encoder
        encoder_input = Input(shape=(128, 128, 3))
        x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_input)
        x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        encoder_output = Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)
        encoder = Model(encoder_input, encoder_output, name='encoder')

        # Quantizer
        # Bu kısımda, vektör kuantizasyonunu uygulamak için özel bir katman yazmanız gerekebilir.
        # Bu örnek için basit bir tam bağlantılı katman kullanacağız.
        quantizer_input = Input(shape=(16, 16, 128))
        quantizer_output = Dense(64, activation='relu')(quantizer_input)
        quantizer = Model(quantizer_input, quantizer_output, name='quantizer')

        # Decoder
        decoder_input = Input(shape=(16, 16, 64))
        x = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(decoder_input)
        x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        decoder_output = Conv2DTranspose(3, 3, activation='sigmoid', strides=2, padding='same')(x)
        decoder = Model(decoder_input, decoder_output, name='decoder')

        # VQ-VAE Model
        vq_vae_input = Input(shape=(128, 128, 3))
        encoded = encoder(vq_vae_input)
        quantized = quantizer(encoded)
        decoded = decoder(quantized)
        vq_vae = Model(vq_vae_input, decoded, name='vq_vae')

        vq_vae.compile(optimizer='adam', loss='mse')
        history = vq_vae.fit(
            train_generator,
            epochs=15,  # Eğitim için epoch sayısı
            validation_data=val_generator
        )
        sample_image = next(iter(train_generator))[0][0]  # Örnek bir görüntü alın

        # Görüntüyü modele verin ve çıktıyı alın
        encoded_image = encoder.predict(np.array([sample_image]))  # Görüntüyü encoder ile kodlayın
        quantized_image = quantizer.predict(encoded_image)  # Kodlanmış görüntüyü quantizer ile dönüştürün
        decoded_image = decoder.predict(quantized_image)  # Dönüştürülmüş görüntüyü decoder ile yeniden oluşturun

        plt.subplot(1, 2, 1)
        plt.title('Görüntü {}'.format(1))
        plt.imshow(sample_image)
        plt.axis('off')  # Eksenleri kapat


        plt.subplot(1, 2, 2)
        plt.title('Görüntü {}'.format(2))
        plt.imshow(decoded_image[0])
        plt.axis('off')  # Eksenleri kapat
        
        plt.show()

    @staticmethod
    def figure_image(images: list) -> None:
        # Orijinal ve yeniden oluşturulmuş görüntüleri gösterin
        plt.figure(figsize=(8, 4))

        for i in range(len(images)):
            plt.subplot(1, len(images), i)
            plt.title('Görüntü {}'.format(i))
            plt.imshow(images[i])
            plt.axis('off')
        plt.show()
        
    @staticmethod
    def save_model(model: Sequential):
        model.save(os.path.join(DLModel.TrainedModelPath, DLModel.TrainedModelName + 'h5'))
    
    @staticmethod
    def generate_image_data(path: str, target_size: tuple, batch_size: int = 32, class_mode: str = 'categorical') -> DirectoryIterator:
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        return train_datagen.flow_from_directory(path, target_size=target_size,batch_size=batch_size,class_mode=class_mode)

    @staticmethod
    def load_created_model(model_file) -> Sequential:
        return load_model(os.path.join(DLModel.TrainedModelPath, model_file))

    @staticmethod
    def augmate_data(image_folder):
        datagen = ImageDataGenerator(
            rotation_range=60,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        for root, _, files in os.walk(image_folder):
            for file in files:
                img_path = os.path.join(root, file)
                img = image.load_img(img_path)
                x = image.img_to_array(img)
                x = x.reshape((1,) + x.shape)
                i = 0
                for batch in datagen.flow(x, batch_size=1):
                    img_augmented = image.array_to_img(batch[0])
                    augmented_file_path = os.path.join(root, f"augmented_{i}_{file}")
                    img_augmented.save(augmented_file_path)
                    i += 1
                    if i >= 5:  # Her örnek için 20 farklı artırılmış veri oluşturabilirsiniz.
                        break

def main():
    dlModel = DLModel()
    dlModel.initialize_model(60, 60)
    dlModel.train_w_tranformer()

if __name__ == '__main__':
    main() 


