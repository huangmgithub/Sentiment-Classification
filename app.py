from flask import Flask, url_for, render_template, redirect, session, Response, jsonify #将内容转换为json
from flask.views import request
import os, json, random, pickle, time
from d3_json import get_json

app = Flask(__name__)

data_path = "./result.pickle"
data = None

@app.route('/index')
def index():
    """
    首页
    :return:
    """
    if session.get('is_login', None):
        # 加载数据
        global data
        with open(data_path, "rb") as f:
            data = pickle.load(f) # {"text", text, "pred", test_pred}
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET','POST'])
def login():
    """
    登录
    :return:
    """
    print('path', request.path)
    print('headers', request.headers)
    print('method', request.method)
    print('url', request.url)
    print('data', request.form)

    if request.method == "POST":
        # username = request.form['username']
        # password = request.form['password']
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        print(username, password)
        if username == "root" and password == "123":
            session['username'] = username
            session['password'] = password
            session['is_login'] = True
            return redirect(url_for('index'))
        else:
            return "Invalid username/password"

    return render_template('login.html')

@app.route('/logout')
def logout():
    """
    注销
    :return:
    """
    session.pop('username',None)
    print('logout')
    return redirect(url_for('login'))

@app.route("/comments", methods=["GET"])
def get_comments():
    # 随机获取一条评论
    result = dict()
    status = False
    message = "No Response"
    try:
        id = random.randint(0, 15000)
        result["id"]= id
        if data:
            result["text"] = data["text"][id] # 评论文本
        status = True
        message = "Response"
    except Exception as e:
        print(e)
    result["status"] = status
    result["message"] = message
    return jsonify(result)

@app.route("/results", methods=["GET", "POST"])
def get_results():
    if request.method == "POST":
        id = int(request.form["id"])
        result = dict()
        status = False
        message = "No Response"
        try:
            pred = data["pred"][id].tolist()  # 预测分类结果
            get_json(pred)  # 分类结果写入json文件以备d3可视化
            status = True
            message = "Response"
        except Exception as e:
            print(e)
        result["status"] = status
        result["message"] = message
        return jsonify(result)

# set the secret key.  keep this really secret:
app.secret_key = os.urandom(24)

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)
