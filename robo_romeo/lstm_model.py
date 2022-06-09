from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.layers import Embedding, LSTM, Add

class LstmModel():
    def __init__(self):

        self.max_caption_length = 36
        self.vocab_size = 7589

        inputs2  = Input(shape=(self.max_caption_length,),name="captions")
        embed_layer = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)

        input_encoded = Input(shape=(8,8,1280),name="images_encoded")
        pooling = GlobalAveragePooling2D()(input_encoded)
        cnn_dense = Dense(256, activation='relu')(pooling)

        combine = Add()([embed_layer,cnn_dense])

        lstm_layer = LSTM(256)(combine)
        decoder = Dense(1000, activation='relu')(lstm_layer)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder)

        model = Model(inputs=[input_encoded, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy' , optimizer='adam',metrics = 'accuracy')

        return model
