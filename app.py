from flask import Flask, request, jsonify
from flask_cors import CORS
import matlab.engine
import random

app = Flask(__name__)
CORS(app)

print("กำลังเชื่อมต่อกับ MATLAB...")
eng = matlab.engine.start_matlab()
print("เชื่อมต่อ MATLAB สำเร็จ!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # 1. รับค่า 6 ตัวแปร หาร 100 และคูณค่าคงที่
        fw = (float(data.get('fw', 0)) / 100) * 16603.48
        pa = (float(data.get('pa', 0)) / 100) * 15821.31
        pf = (float(data.get('pf', 0)) / 100) * 32763.28
        t  = (float(data.get('t', 0)) / 100)  * 21733.86
        w  = (float(data.get('w', 0)) / 100)  * 16920.80
        lr = (float(data.get('lr', 0)) / 100) * 29259.83
        
        # 2. สุ่มค่า M (Moisture) ระหว่าง 60 ถึง 65
        m_random = round(random.uniform(60, 65), 2)
        
        # 3. ส่งข้อมูลเข้า MATLAB
        eng.workspace['input_data'] = eng.struct({
            'E_fw': fw, 'E_pa': pa, 'E_pf': pf, 
            'E_T': t, 'E_W': w, 'E_Lr': lr, 'M': m_random
        })
        eng.eval("T = struct2table(input_data)", nargout=0)
        
        # 4. โหลดโมเดล
        eng.eval("S = load('RegressionLearnerSession2.mat')", nargout=0)
        eng.eval("names = fieldnames(S); modelVar = S.(names{1});", nargout=0)
        
        # 5. สั่งคำนวณ
        result = eng.eval("modelVar.predictFcn(T)", nargout=1)
        hhv_value = float(result)

        # ==========================================
        # ส่วนที่เพิ่ม: แสดงค่าใน Terminal
        # ==========================================
        print("\n" + "="*30)
        print(f" ผลการคำนวณ HHV: {hhv_value:.2f} kJ/kg")
        print(f" ค่าความชื้น (M) ที่ใช้: {m_random}%")
        print("="*30 + "\n")
        
        return jsonify({
            'success': True, 
            'hhv': hhv_value,
            'random_m': m_random 
        })

    except Exception as e:
        print(f" Error logic: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=False)