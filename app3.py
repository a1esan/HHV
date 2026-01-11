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
        # รับค่า 6 ตัวแปรให้ตรงกับใน HTML
        fw_v = float(data.get('fw', 0))
        pa_v = float(data.get('pa', 0))
        pf_v = float(data.get('pf', 0))
        t_v  = float(data.get('t', 0))
        w_v  = float(data.get('w', 0))
        lr_v = float(data.get('lr', 0))

        # โหลดโมเดล (ต้องทำใน MATLAB Export ก่อนนะ)
        eng.eval("S = load('RegressionLearnerSession2.mat')", nargout=0)
        eng.eval("names = fieldnames(S); modelVar = S.(names{1});", nargout=0)

        # ฟังก์ชันคำนวณ (ใช้ชื่อ Fw, Pa, Pf, T, W, Lr ตามที่คุณตั้งไว้ใน MATLAB)
        eng.workspace['input_data'] = eng.struct({
            'Fw': fw_v, 'Pa': pa_v, 'Pf': pf_v, 'T': t_v, 'W': w_v, 'Lr': lr_v
        })
        eng.eval("T_table = struct2table(input_data)", nargout=0)
        
        # คำนวณผลลัพธ์
        result = eng.eval("modelVar.predictFcn(T_table)", nargout=1)

        return jsonify({'success': True, 'hhv': float(result)})

    except Exception as e:
        print(f"Error logic: {str(e)}") # สังเกต Error ที่นี่ครับ
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=False)