from flask import Flask
from flask_restx import Api, Resource, fields
from model_runner import run_model
from intent_classifier import classify_text

app = Flask(__name__)
api = Api(app, version='1.1', title='반도체 검사 시스템 인터페이스 API 테스트',
          description='LLM 명령 실행 및 Intent 분류')


# LLM 명령 실행 namespace
ns_instruct = api.namespace('instruct', description='LLM 명령 실행')

input_model = api.model('InputModel', {
    'text': fields.String(required=True, description='명령어 텍스트')
})

gpu_model = api.model('GpuModel', {
    'allocated_mb': fields.Float(description='사용된 GPU 메모리'),
    'reserved_mb': fields.Float(description='예약된 GPU 메모리')
})

output_model = api.model('OutputModel', {
    'output': fields.String(description='LLM 응답 함수 호출 결과'),
    'elapsed_time': fields.Float(description='응답 시간(초)'),
    'gpu_memory': fields.Nested(gpu_model)
})

@ns_instruct.route('/')
class Instruct(Resource):
    @ns_instruct.expect(input_model)
    @ns_instruct.marshal_with(output_model)
    def post(self):
        """사용자 명령어를 입력받아 함수 호출 형식으로 응답"""
        user_input = api.payload['text']
        return run_model(user_input)


#Intent 분류 namespace
ns_classify = api.namespace('classify', description='문장 intent 분류')

classify_input_model = api.model('ClassifyInput', {
    'text': fields.String(required=True, description='분류할 문장')
})

classify_output_model = api.model('ClassifyOutput', {
    'label': fields.String(description='예측된 라벨'),
    'score': fields.Float(description='예측 확률')
})

@ns_classify.route('/')
class Classify(Resource):
    @ns_classify.expect(classify_input_model)
    @ns_classify.marshal_with(classify_output_model)
    def post(self):
        """문장을 분류하고 intent 라벨을 반환합니다."""
        text = api.payload['text']
        return classify_text(text)

# 서버 실행

if __name__ == '__main__':
    app.run(debug=True)
