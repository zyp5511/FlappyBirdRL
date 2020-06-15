import argparse
import torch
import os

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing, dump_pickle


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models/double_train")
    parser.add_argument("--num_test", type=int, default=50)

    args = parser.parse_args()
    return args


def test(opt, model_dir):
    frame_number = 0 
    action_sum = 0
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/flappy_bird_2000000".format(model_dir))
    else:
        model = torch.load("{}/flappy_bird".format(opt.saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    game_state = FlappyBird()
    image, reward, terminal, score = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    flag = True
    while flag:
        frame_number += 1
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        action_sum += action
        next_image, reward, terminal, score = game_state.next_frame(action)
        if terminal or score == 50:
            flag = False
            break
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        state = next_state
      
    return score, frame_number, action_sum

if __name__ == "__main__":
    opt = get_args()
    result = "/media/yipeng/data/FlappyBirdRL/inference_avg_click"
    model_dir = "/media/yipeng/data/FlappyBirdRL/trained_models"
    for model_fn in sorted(os.listdir(model_dir)):
        if not model_fn.startswith("s"):
            continue
        print(model_fn)
        score_list = []
        frame_list = []
        action_list = []
        for i in range(opt.num_test):
            score, frame_number, action_sum = test(opt, os.path.join(model_dir,model_fn))
            score_list.append(score)
            frame_list.append(frame_number)
            action_list.append(action_sum)
            print(i, score, frame_number, action_sum )
        fn = model_fn.split("/")[-1] +".pkl"
        res = {"score":score_list, "frame":frame_list, "action":action_list}
        dump_pickle(os.path.join(result,fn),res)
