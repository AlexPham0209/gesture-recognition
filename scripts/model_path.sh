#!/bin/bash
MODEL_URL=https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
MODEL_PATH=model

mkdir model
wget -q $MODEL_URL -P $MODEL_PATH