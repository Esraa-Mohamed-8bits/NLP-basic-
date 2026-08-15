"""
Simple app: type in a 3D vector, pick an axis and an angle,
and see the rotated vector.
"""

import streamlit as st
import numpy as np

st.title("3D Vector Rotation")

st.write("Enter a vector (x, y, z), choose an axis, and choose an angle to rotate it.")

# get the vector from the user
x = st.number_input("x", value=1.0)
y = st.number_input("y", value=0.0)
z = st.number_input("z", value=0.0)
vector = np.array([x, y, z])

# pick which axis to rotate around
axis = st.selectbox("Axis of rotation", ["x", "y", "z"])

# angle in degrees, we convert to radians for the math
theta_deg = st.slider("Angle (theta) in degrees", 0, 360, 90)
theta = np.radians(theta_deg)


def rotate_x(v, theta):
    # rotation matrix around the x-axis
    R = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return R @ v


def rotate_y(v, theta):
    # rotation matrix around the y-axis
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return R @ v


def rotate_z(v, theta):
    # rotation matrix around the z-axis
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R @ v


# pick the right rotation function based on the axis chosen
if axis == "x":
    rotated_vector = rotate_x(vector, theta)
elif axis == "y":
    rotated_vector = rotate_y(vector, theta)
else:
    rotated_vector = rotate_z(vector, theta)

st.subheader("Result")
st.write("Original vector:", vector)
st.write(f"Rotated vector (around {axis}-axis by {theta_deg}°):", rotated_vector)
