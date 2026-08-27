echo "Installing the necessary packages ..."
pip install -r script/requirements.txt

echo "Installing pytorch3d ..."
# cd third_party/pytorch3d_simplified
# pip install -e .
# cd ../..
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# location of sapien, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/sapien"
SAPIEN_LOCATION=$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien
# Adjust some code in wrapper/urdf_loader.py
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + ".srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER

# Verify the sapien patch was applied
if ! grep -q 'encoding="utf-8"' "$URDF_LOADER" 2>/dev/null; then
    echo "WARNING: sapien urdf_loader.py patch may not have been applied. Check sapien version."
fi


echo "Adjusting code in mplib/planner.py ..."
# location of mplib, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/mplib"
MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib

# Adjust some code in planner.py
# Note: This removes the 'or collide' check from the screw plan condition.
# This is intentional: mplib's collision check can be overly conservative
# in cluttered table scenes, causing valid plans to fail. Removing it
# allows the planner to find paths that would otherwise be rejected.
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

# Verify the mplib patch was applied
if grep -q 'or collide' "$PLANNER" 2>/dev/null; then
    echo "WARNING: mplib planner.py patch may not have been applied. Check mplib version."
fi

echo "Installing Curobo ..."
cd envs
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../..

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download asserts from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See INSTALLATION.md for more instructions."
