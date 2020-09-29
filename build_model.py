#--------------------Model-----------------------
def build_model(seq_len=107, pred_len=68, dropout=0.5,sp_dropout=0.2,hidden_dim=256, n_layers=3):
    inputs = L.Input(shape=(seq_len,3))
    embed = L.Embedding(input_dim=14, output_dim=200)(inputs)
    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)
    for x in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    truncated = hidden[:,:pred_len]
    outputs = L.Dense(5, activation='linear')(truncated)
    model = tf.keras.Model(inputs=inputs ,outputs=outputs)
    model.compile(optimizer=tf.optimizers.Adam(), loss=custom_score)