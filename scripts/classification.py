#!/usr/bin/env python3
"""Classify velocity and angular velocity data based on machine learning models."""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("learning_source/velocity.csv")
X = df[['Body', 'Head', 'UpperBody', 'LeftArm1', 'LeftArm2', 'RightArm1', 'RightArm2', 'LeftLeg1', 'LeftLeg2', 'RightLeg1', 'RightLeg2', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Body_translational']]
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
accuracy_score = accuracy_score(y_test, pred)
print(f"Classification accuracy: {accuracy_score:.2f}")
for i in range(len(pred)):
    print(f"Sample {i}: Predicted label = {pred[i]}, True label = {y_test.iloc[i]}")