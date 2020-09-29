def gru_layer(hidden_dim, dropout):
    gru_content = L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal')
    gru_content = L.Bidirectional(gru_content)
    return gru_content