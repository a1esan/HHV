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
        # 1. รับค่า % องค์ประกอบพื้นฐาน
        fw_v = float(data.get('fw', 0))
        pa_v = float(data.get('pa', 0))
        pf_v = float(data.get('pf', 0))
        t_v  = float(data.get('t', 0))
        w_v  = float(data.get('w', 0))
        lr_v = float(data.get('lr', 0))
        
        # 2. โหลดโมเดลเตรียมไว้
        eng.eval("S = load('RegressionLearnerSession2.mat')", nargout=0)
        eng.eval("names = fieldnames(S); modelVar = S.(names{1});", nargout=0)

        # 3. สร้างฟังก์ชันภายในเพื่อคำนวณแยกทีละก้อน
        def calc_single(fw, pa, pf, t, w, lr):
            eng.workspace['input_data'] = eng.struct({
                'Fw': fw, 'Pa': pa, 'Pf': pf, 'T': t, 'W': w, 'Lr': lr
            })
            eng.eval("T_single = struct2table(input_data)", nargout=0)
            return float(eng.eval("modelVar.predictFcn(T_single)", nargout=1))

        # 4. คำนวณ HHV แยกตามองค์ประกอบ แล้วคูณตัวเลขที่ต้องการ
        # (บรรทัด +get_hhv ที่เคยงง ผมลบออกแล้วใช้แบบนี้แทนครับ)
        hhv_fw = calc_single(fw_v, 0, 0, 0, 0, 0) * 3968.3276
        hhv_pa = calc_single(0, pa_v, 0, 0, 0, 0) * 3781.3831
        hhv_pf = calc_single(0, 0, pf_v, 0, 0, 0) * 7830.6109
        hhv_t  = calc_single(0, 0, 0, t_v, 0, 0) * 5194.5177
        hhv_w  = calc_single(0, 0, 0, 0, w_v, 0) * 4044.1689
        hhv_lr = calc_single(0, 0, 0, 0, 0, lr_v) * 6993.2662

        # 5. รวมค่าทั้งหมดเข้าด้วยกัน
        total_hhv = hhv_fw + hhv_pa + hhv_pf + hhv_t + hhv_w + hhv_lr

        return jsonify({'success': True, 'hhv': total_hhv})

    except Exception as e:
        print(f"Error logic: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=False)