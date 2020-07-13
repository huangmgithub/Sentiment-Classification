import json
from collections import defaultdict

c = ["位置", "服务", "价格", "环境", "菜品", "其他"]

c_pos = ["交通是否便利", "距离商圈远近", "是否容易寻找"]
c_ser = ["排队等候时间", "服务人员态度", "是否容易停车", "点菜/上菜速度"]
c_pri = ["价格水平", "性价比", "折扣力度"]
c_env = ["装修情况", "嘈杂情况", "就餐空间", "卫生情况"]
c_dis = ["分量", "口感", "外观", "推荐程度"]
c_oth = ["本次消费感受", "再次消费的意愿"]

def get_json(a):
    r = defaultdict(list)
    n = 0
    r["位置"] = c_pos
    r["服务"] = c_ser
    r["价格"] = c_pri
    r["环境"] = c_env
    r["菜品"] = c_dis
    r["其他"] = c_oth

    res = dict()
    res["name"] = "类别"
    res["children"] = []
    for i, j in r.items():
        r1 = dict()
        r1["name"] = i
        r1["children"] = []
        for k in j:
            r2  = dict()
            r2["name"] = k + str(int(a[n]))
            n += 1
            r2["size"] = 300
            r1["children"].append(r2)
        res["children"].append(r1)


    with open("./static/d3/comment_style.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)

if __name__ == "__main__":
    a = list(range(20))
    get_json(a)



