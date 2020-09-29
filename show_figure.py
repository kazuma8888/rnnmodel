def show_figure(history):
    fig = px.line(
        history.history, y=['loss', 'val_loss'],
        labels={'index': 'epoch', 'value': 'Score'}, 
        title='Training History')
    fig.show()