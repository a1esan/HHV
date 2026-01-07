from flask import Flask, request, jsonify
from flask_cors import CORS
import matlab.engine

app = Flask(__name__)
CORS(app)

print("กำลังเชื่อมต่อกับ MATLAB...")
eng = matlab.engine.start_matlab()
print("เชื่อมต่อ MATLAB สำเร็จ!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # รับค่า 3 ตัวแปร
        fw = float(data.get('fw', 0))
        pa = float(data.get('pa', 0))
        pf = float(data.get('pf', 0))
        
        # ส่งข้อมูลเข้า MATLAB
        eng.workspace['input_data'] = eng.struct({
            'Fw': fw, 'Pa': pa, 'Pf': pf
        })
        eng.eval("T = struct2table(input_data)", nargout=0)
        
        # โหลดไฟล์โมเดล
        eng.eval("S = load('RegressionLearnerSession02.mat')", nargout=0)
        
        # บรรทัดนี้คือเคล็ดลับ: มันจะไปดึงโมเดลออกมาไม่ว่าข้างในจะชื่ออะไร
        eng.eval("names = fieldnames(S); modelVar = S.(names{1});", nargout=0)
        
        # สั่งคำนวณ
        result = eng.eval("modelVar.predictFcn(T)", nargout=1)
        
        return jsonify({'success': True, 'hhv': float(result)})
    except Exception as e:
        print(f"Error logic: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=False)