# Data analysis
- Document here the project: robo_romeo
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for robo_romeo in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/robo_romeo`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "robo_romeo"
git remote add origin git@github.com:{group}/robo_romeo.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
robo_romeo-run
```

# Install

Go to `https://github.com/{group}/robo_romeo` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/robo_romeo.git
cd robo_romeo
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
robo_romeo-run
```
# Attention Model

<img width="716" alt="Screenshot 2022-06-21 at 17 34 46" src="https://user-images.githubusercontent.com/103648207/174852206-2bf930da-ae4c-4293-bb1a-7818eaa1ab00.png">
<img width="615" alt="Screenshot 2022-06-21 at 17 35 26" src="https://user-images.githubusercontent.com/103648207/174852319-342c0405-ee32-453c-bb2d-09981d645493.png">
