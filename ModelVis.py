from keras.utils.vis_utils import plot_model
import Models

def draw_graph(model, filename):
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=False)

def main():
    factory = Models.ModelFactory(10)
    draw_graph(factory.make_basic_model(), "basic.png")
    draw_graph(factory.make_vgg_model_2_block(), "vgg2.png")
    draw_graph(factory.make_vgg_model_3_block(), "vgg3.png")
    draw_graph(factory.make_vgg_model_4_block(), "vgg4.png")

if __name__ == "__main__":
    main()