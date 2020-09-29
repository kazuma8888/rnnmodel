#----------------------KFold------------------------
def CrossValidation(x, y, public_inputs, private_inputs):
    k = 5
    kf = KFold(n_splits=k)
    A = pd.DataFrame(index=[], columns=[])
    model = build_model()
    model.summary()
    for e, (train_index, eval_index) in enumerate(kf.split(x, y)):
            x_train = x[train_index]
            y_train = y[train_index]
            x_eval = x[eval_index]
            y_eval = y[eval_index]
            model = build_model()
            history = model.fit( x_train, y_train,
                                 validation_data=(x_eval, y_eval),
                                 batch_size=64,
                                 epochs=1,
                                 verbose=2,
                                 callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=5),
                                            tf.keras.callbacks.ModelCheckpoint('model.h5') #checkpointで保存する
                                           ]
                                )
            show_figure(history)
            
            model_public = build_model(seq_len=107, pred_len=107)
            model_public.load_weights('model.h5')
            public_preds = model_public.predict(public_inputs)
            
            model_private = build_model(seq_len=130, pred_len=130)
            model_private.load_weights('model.h5')
            private_preds = model_private.predict(private_inputs)

            pre_postprocess(public_df, private_df, public_preds, private_preds, k, e, A)
    return A