import random
from inter import identify


def game(image_path):
    player_gesture = identify(image_path)

    gestures = ['rock', 'paper', 'scissors']
    computer_gesture = random.choice(gestures)

    if player_gesture == computer_gesture:
        result = "Draw!"
    elif (player_gesture == 'rock' and computer_gesture == 'scissors') or \
            (player_gesture == 'scissors' and computer_gesture == 'paper') or \
            (player_gesture == 'paper' and computer_gesture == 'rock'):
        result = "You Win!"
    else:
        result = "You Lose!"

    return player_gesture, computer_gesture, result