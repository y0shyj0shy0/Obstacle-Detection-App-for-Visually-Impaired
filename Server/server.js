
const express = require('express');
const cors = require('cors');
const session = require('express-session');
const bodyParser = require('body-parser');
const mysql = require('mysql');
const multer = require('multer');  // multer 추가
const path = require('path'); // 파일 경로 처리
const uuid4 = require('uuid4');
const app = express();
let list = [];
let randomID = 1;
// CORS 설정
app.use(cors());
app.use(express.json());
app.use(bodyParser.json())
// 세션 설정
app.use(session({
    resave: false,
    saveUninitialized: false,
    secret: 'mySession',
    cookie: {
        httpOnly: true,
        secure: false, // 개발환경에서는 false로 설정
    },
}));

// // MySQL 연결
// const db = mysql.createConnection({
//     user: "root",
//     host: "localhost",
//     password: "5721",
//     database: "army_db"
// });

const hostname = "59.19.113.82";
const port = 3003;

// db.connect((error) => {
//     if (error) console.log(error);
//     else console.log('db connected');
// });

// Multer 설정 - 이미지 파일 업로드 처리
const upload = multer({
  storage: multer.diskStorage({
    filename(req, file, done) {
        const ext = path.extname(file.originalname);
        const filename = randomID + ext;
        //console.log(file);
        done(null, filename);
        
    },
    destination(req, file, done) {
        //console.log(file);
        done(null, path.join(__dirname, 'uploads'));
    },

  }),
  
});

// 'uploads' 폴더를 static으로 제공
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// 기본 엔드포인트
app.get('/', (req, res) => {
    res.json({d: 'hello123'});
});

// 이미지 파일 업로드 처리 엔드포인트
app.post('/upload', upload.single('file'), (req, res) => {
    if (req.file) {
        //console.log('Uploaded file:', req.file);
        // 파일 저장 후 DB에 관련 정보를 저장하거나 추가 작업을 할 수 있습니다.
        res.json({ message: 'File uploaded successfully', file: req.file });
    } else {
        res.status(400).json({ message: 'No file uploaded' });
    }
});

app.get('/getTTS', (req, res) => {
    res.json({tts: list.pop()})
});

app.post('/tts', (req, res) => {
    if (req.body) {  // req.body로 변경
        console.log('TTS Data:', req.body.tts);  // 전달된 데이터 출력
        res.json({ message: 'TTS request received', receivedData: req.body});  // 클라이언트로 응답
        console.log('리스트길이: ', list.length);
        if (list.length > 0) {
            if (list[0] !== req.body.tts) {
                // tts 값이 다를 때 list 갱신
                list.pop();
                list.push(req.body.tts);
            } else {
                // tts 값이 같을 때 list 비우기
                list.pop();
            }
        } else {
            // list가 비어있을 때 tts 추가
            list.push(req.body.tts);
        }


    } else {
        res.status(400).json({ message: 'No data received' });  // 데이터가 없을 경우
    }
});

// 서버 실행
app.listen(port, hostname, () => {
    console.log('running on port 3003');
});
