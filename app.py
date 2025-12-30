from flask import Flask, request, jsonify
from flask_cors import CORS
import matlab.engine

app = Flask(__name__)
CORS(app) # อนุญาตให้หน้าเว็บส่งข้อมูลมาหา Python ได้

# เริ่มรัน MATLAB Engine (ตอนรันครั้งแรกจะใช้เวลาประมาณ 10-20 วินาที)
print("กำลังเชื่อมต่อกับ MATLAB...")
eng = matlab.engine.start_matlab()

# สั่งให้ MATLAB โหลดไฟล์ Model ของคุณ
# **สำคัญ: ไฟล์ hhv_model.mat ต้องอยู่ในโฟลเดอร์เดียวกับไฟล์นี้**
eng.eval("load('RegressionLearnerSession02.mat')", nargout=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        fw, pa, pf = float(data['fw']), float(data['pa']), float(data['pf'])
        
        # เตรียมตารางข้อมูล
        eng.workspace['input_data'] = eng.struct({'Fw': fw, 'Pa': pa, 'Pf': pf})
        eng.eval("T = struct2table(input_data)", nargout=0)
        
        # โหลดไฟล์และดึงตัวโมเดลออกมา
        eng.eval("S = load('RegressionLearnerSession02.mat')", nargout=0)
        eng.eval("names = fieldnames(S); modelVar = S.(names{1});", nargout=0)
        
        # สั่งคำนวณผ่านตัวแปรที่ดึงออกมา
        result = eng.eval("modelVar.predictFcn(T)", nargout=1)
        
        return jsonify({'success': True, 'hhv': float(result)})
    except Exception as e:
        print(f"Error logic: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # รันเซิร์ฟเวอร์ที่เครื่องคุณ Port 5000
    app.run(port=5000)