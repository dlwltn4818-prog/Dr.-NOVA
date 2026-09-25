"""
Clinical Diagnostic AI Agent - Production Standard EMR Engine
- Universal 26 Clinical Specialties Coverage
- Clinical Freeze Logic (Hides diagnosis card & freezes confidence score on administrative queries)
- Privacy-Preserving UI (Standard 'ex)' placeholders, zero private data hardcoding)
- Auto-Expanding Multiline Textarea (Gemini-style dynamic height adjustment)
- Chat Edit / Delete with Real-time DB Sync
- Patient ID-based Longitudinal Record Tracking (SQLite3)
"""
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import uvicorn

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FREE_MODEL_POOL = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest"
]

DB_FILE = "clinical_records.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                patient_name TEXT,
                birth_date TEXT,
                biological_sex TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encounters (
                encounter_id TEXT PRIMARY KEY,
                patient_id TEXT,
                encounter_seq INTEGER,
                chief_complaint TEXT,
                created_at TEXT,
                history_json TEXT,
                diagnosis_summary TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        conn.commit()

init_db()

app = FastAPI(title="Clinical AI EMR & Diagnostic Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
당신은 대한민국 상급종합병원(대학병원) 수준의 통합 임상진료 전문의 AI입니다.
내과, 외과, 정형외과, 신경과, 재활의학과, 이비인후과 등 26개 전체 전문 진료과목의 질환을 감별 진단합니다.

[과거 병력 참조 원칙]
- 환자의 누적 병력은 기저 위험도(면역 저하 여부, 약물 상호작용 등)를 파악하는 참고자료입니다.
- 금일 호소하는 새로운 증상이 기저질환과 완전히 무관한 독립적 급성 질환(예: 급성 감염병, 외상, 타 분과 질환)일 가능성을 열어두고 객관적으로 감별하십시오. 과거 진단명에 맹목적으로 얽매여 새로운 원인 질환을 배제하는 우를 범하지 마십시오.

[⚠️ 임상 확신도 및 진단서 발행 절대 규칙 (위반 엄격 금지)]
1. [비증상성/행정 질문 시 진단서 발행 절대 금지]:
   - 환자가 비용("얼마인가요?", "비용은 어느 정도 나와요?"), 시간/일정("얼마나 걸리나요?"), 예약, 병원 위치, 단순 동의/맞장구 등 '비의학적 행정 질문'을 할 경우:
     👉 `diagnosis_report` 필드는 반드시 `null`로 반환하십시오. 절대로 진단서나 확신도 점수를 출력하지 마십시오.
     👉 의사의 message로 원무 수납, 급여/비급여, 실손보험, 검사 예약 절차 등을 성심껏 설명하고, 검사 진행 여부를 묻는 선택지(chips)를 제공하십시오.
     👉 rationale에는 '비증상성 행정/원무 문의 - 의학적 단서 불변으로 진단서 미발행'이라고 명시하십시오.
2. [정밀 임상 진단서 발행 및 점수 변동 기준]:
   - `diagnosis_report`는 오직 환자가 '새로운 증상 양상, 부위, 발작 빈도, 기저질환 단서' 등 의학적 단서를 실질적으로 추가 제공했을 때만 생성하거나 업데이트하십시오.
   - 단서가 추가되지 않았는데 확신도 점수가 오르거나(75%->85%) 떨어지는(75%->65%) 등의 점수 왜곡은 임상적 오류이므로 엄격히 금지합니다.
3. [행동 유형]:
   - 일반 문진: "ASK_QUESTION"
   - 검사 처방: "ORDER_TEST"
   - 확정 진단: "DIAGNOSE"
   - 응급 신호: "ESCALATE"

[출력 JSON 규격 - 순수 JSON 포맷만 반환]
{
  "action_type": "ASK_QUESTION" | "ORDER_TEST" | "ESCALATE" | "DIAGNOSE",
  "message": "환자에게 건넬 친절하고 전문적인 의사의 질문 또는 설명",
  "rationale": "임상적/행정적 판단 이유",
  "recommended_department": "추천 전문과 (예: 신경과, 정형외과, 순환기내과 등)",
  "diagnosis_report": null 또는 {
    "primary_diagnosis": "정밀 의학 진단명",
    "differential_diagnoses": ["감별 대상 질환 1", "감별 대상 질환 2"],
    "confidence_score": 75,
    "clinical_reasoning": "의학적 판단 요약"
  },
  "chips": ["선택지 1", "선택지 2"]
}
"""

def clean_and_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)

def run_agent_reasoning(patient_info: Dict[str, Any], past_encounters: List[Dict[str, Any]], current_encounter: Dict[str, Any]) -> Dict[str, Any]:
    past_summary = ""
    for enc in past_encounters:
        past_summary += f"- [{enc['created_at']} 제{enc['encounter_seq']}차] 주호소: {enc['chief_complaint']} / 최종진단: {enc.get('diagnosis_summary', '문진 진행')}\n"

    current_dialogue = ""
    for turn in current_encounter["history"]:
        role_label = "환자" if turn["role"] == "user" else "AI 의사"
        current_dialogue += f"{role_label}: {turn['content']}\n"

    full_prompt = f"""{SYSTEM_PROMPT}

[환자 인적사항]
- 환자번호: {patient_info['patient_id']}
- 성함: {patient_info['patient_name']} (생년월일: {patient_info['birth_date']}, 성별: {patient_info['biological_sex']})

[과거 누적 병력]
{past_summary if past_summary else "(신규 등록 환자 - 이전 병력 없음)"}

[금일 제{current_encounter['encounter_seq']}차 진료 진행 내역]
- 초기 주호소: {current_encounter['chief_complaint']}
- 대화 내역:
{current_dialogue if current_dialogue else "(초기 문진 시작)"}

[지시]
환자의 방금 마지막 발언이 '비용, 시간, 예약 등 행정 질문'인지 '의학적 증상'인지 철저히 판단하십시오. 행정 질문일 때는 diagnosis_report를 반드시 null로 처리하고 진단서 출력을 차단하십시오. JSON으로만 응답하십시오.
"""

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }

    last_err = ""
    for model_name in FREE_MODEL_POOL:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        try:
            res = requests.post(api_url, json=payload, timeout=15)
            res_data = res.json()
            if "candidates" in res_data:
                raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return clean_and_parse_json(raw_text)
            err_msg = res_data.get("error", {}).get("message", "Error")
            last_err = err_msg
            time.sleep(0.5)
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5)

    raise HTTPException(status_code=500, detail=f"진단 엔진 지연: {last_err}")

class PatientLookupRequest(BaseModel):
    patient_id: str
    patient_name: str
    birth_date: str
    biological_sex: str

class StartEncounterRequest(BaseModel):
    patient_id: str
    chief_complaint: str

class ChatAnswerRequest(BaseModel):
    answer: str

class EditChatRequest(BaseModel):
    index: int
    new_text: str

class DeleteChatRequest(BaseModel):
    index: int

@app.post("/api/patients/auth")
def authenticate_or_register_patient(req: PatientLookupRequest):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT patient_id, patient_name, birth_date, biological_sex, created_at FROM patients WHERE patient_id = ?", (req.patient_id,))
        row = cursor.fetchone()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        if not row:
            cursor.execute("INSERT INTO patients VALUES (?, ?, ?, ?, ?)", (req.patient_id, req.patient_name, req.birth_date, req.biological_sex, now_str))
            conn.commit()
            patient_data = req.model_dump()
        else:
            patient_data = {"patient_id": row[0], "patient_name": row[1], "birth_date": row[2], "biological_sex": row[3], "created_at": row[4]}
        
        cursor.execute("SELECT encounter_id, encounter_seq, chief_complaint, created_at, diagnosis_summary FROM encounters WHERE patient_id = ? ORDER BY encounter_seq DESC", (req.patient_id,))
        enc_rows = cursor.fetchall()
        encounters = [{"encounter_id": r[0], "encounter_seq": r[1], "chief_complaint": r[2], "created_at": r[3], "diagnosis_summary": r[4]} for r in enc_rows]
        return {"patient": patient_data, "encounters": encounters}

@app.get("/api/encounters/{encounter_id}")
def get_encounter(encounter_id: str):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT encounter_id, patient_id, encounter_seq, chief_complaint, created_at, history_json, diagnosis_summary FROM encounters WHERE encounter_id = ?", (encounter_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="진료 기록 없음")
        return {
            "encounter_id": row[0],
            "patient_id": row[1],
            "encounter_seq": row[2],
            "chief_complaint": row[3],
            "created_at": row[4],
            "history": json.loads(row[5]),
            "diagnosis_summary": row[6]
        }

@app.post("/api/encounters/start")
def start_new_encounter(req: StartEncounterRequest):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT patient_id, patient_name, birth_date, biological_sex FROM patients WHERE patient_id = ?", (req.patient_id,))
        p_row = cursor.fetchone()
        if not p_row:
            raise HTTPException(status_code=404, detail="환자 정보 없음")
        patient_info = {"patient_id": p_row[0], "patient_name": p_row[1], "birth_date": p_row[2], "biological_sex": p_row[3]}

        cursor.execute("SELECT encounter_id, encounter_seq, chief_complaint, created_at, diagnosis_summary FROM encounters WHERE patient_id = ? ORDER BY encounter_seq ASC", (req.patient_id,))
        past_encounters = [{"encounter_seq": r[1], "chief_complaint": r[2], "created_at": r[3], "diagnosis_summary": r[4]} for r in cursor.fetchall()]

        next_seq = len(past_encounters) + 1
        encounter_id = f"enc-{uuid.uuid4().hex[:6]}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        current_encounter = {
            "encounter_id": encounter_id,
            "patient_id": req.patient_id,
            "encounter_seq": next_seq,
            "chief_complaint": req.chief_complaint,
            "created_at": now_str,
            "history": []
        }

        action = run_agent_reasoning(patient_info, past_encounters, current_encounter)
        current_encounter["history"].append({
            "role": "model",
            "content": action["message"],
            "action": action
        })

        diag_init = action.get("diagnosis_report", {}).get("primary_diagnosis", "문진 진행 중") if action.get("diagnosis_report") else "문진 진행 중"
        cursor.execute("INSERT INTO encounters VALUES (?, ?, ?, ?, ?, ?, ?)", 
                       (encounter_id, req.patient_id, next_seq, req.chief_complaint, now_str, json.dumps(current_encounter["history"], ensure_ascii=False), diag_init))
        conn.commit()

        return {"encounter_id": encounter_id, "next_action": action}

@app.post("/api/encounters/{encounter_id}/respond")
def respond_encounter(encounter_id: str, req: ChatAnswerRequest):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT encounter_id, patient_id, encounter_seq, chief_complaint, created_at, history_json, diagnosis_summary FROM encounters WHERE encounter_id = ?", (encounter_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="진료 세션 오류")
        
        patient_id = row[1]
        current_encounter = {
            "encounter_id": row[0],
            "patient_id": patient_id,
            "encounter_seq": row[2],
            "chief_complaint": row[3],
            "created_at": row[4],
            "history": json.loads(row[5]),
            "diagnosis_summary": row[6]
        }

        cursor.execute("SELECT patient_id, patient_name, birth_date, biological_sex FROM patients WHERE patient_id = ?", (patient_id,))
        p_row = cursor.fetchone()
        patient_info = {"patient_id": p_row[0], "patient_name": p_row[1], "birth_date": p_row[2], "biological_sex": p_row[3]}

        cursor.execute("SELECT encounter_id, encounter_seq, chief_complaint, created_at, diagnosis_summary FROM encounters WHERE patient_id = ? AND encounter_id != ? ORDER BY encounter_seq ASC", (patient_id, encounter_id))
        past_encounters = [{"encounter_seq": r[1], "chief_complaint": r[2], "created_at": r[3], "diagnosis_summary": r[4]} for r in cursor.fetchall()]

        current_encounter["history"].append({"role": "user", "content": req.answer})
        action = run_agent_reasoning(patient_info, past_encounters, current_encounter)
        current_encounter["history"].append({
            "role": "model",
            "content": action["message"],
            "action": action
        })

        diag_summary = current_encounter["diagnosis_summary"]
        if action.get("diagnosis_report") and action["diagnosis_report"].get("primary_diagnosis"):
            d_rep = action["diagnosis_report"]
            p_name = d_rep.get("primary_diagnosis", "")
            c_score = d_rep.get("confidence_score", 0)
            if c_score >= 50:
                diag_summary = f"{p_name} ({c_score}%)"
            else:
                diag_summary = "미상 (추가 검사 필요)"
                
        cursor.execute("UPDATE encounters SET history_json = ?, diagnosis_summary = ? WHERE encounter_id = ?", 
                       (json.dumps(current_encounter["history"], ensure_ascii=False), diag_summary, encounter_id))
        conn.commit()

        return action

@app.put("/api/encounters/{encounter_id}/chat/edit")
def edit_chat_message(encounter_id: str, req: EditChatRequest):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT patient_id, encounter_seq, chief_complaint, created_at, history_json, diagnosis_summary FROM encounters WHERE encounter_id = ?", (encounter_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="진료 세션 없음")
        
        patient_id = row[0]
        history = json.loads(row[4])
        if req.index < 0 or req.index >= len(history):
            raise HTTPException(status_code=400, detail="유효하지 않은 메시지 번호")

        history = history[:req.index + 1]
        history[req.index]["content"] = req.new_text

        cursor.execute("SELECT patient_id, patient_name, birth_date, biological_sex FROM patients WHERE patient_id = ?", (patient_id,))
        p_row = cursor.fetchone()
        patient_info = {"patient_id": p_row[0], "patient_name": p_row[1], "birth_date": p_row[2], "biological_sex": p_row[3]}

        cursor.execute("SELECT encounter_id, encounter_seq, chief_complaint, created_at, diagnosis_summary FROM encounters WHERE patient_id = ? AND encounter_id != ? ORDER BY encounter_seq ASC", (patient_id, encounter_id))
        past_encounters = [{"encounter_seq": r[1], "chief_complaint": r[2], "created_at": r[3], "diagnosis_summary": r[4]} for r in cursor.fetchall()]

        current_encounter = {
            "encounter_id": encounter_id,
            "patient_id": patient_id,
            "encounter_seq": row[1],
            "chief_complaint": row[2],
            "created_at": row[3],
            "history": history
        }

        new_action = run_agent_reasoning(patient_info, past_encounters, current_encounter)
        history.append({
            "role": "model",
            "content": new_action["message"],
            "action": new_action
        })

        diag_summary = new_action.get("diagnosis_report", {}).get("primary_diagnosis", row[5]) if new_action.get("diagnosis_report") else row[5]

        cursor.execute("UPDATE encounters SET history_json = ?, diagnosis_summary = ? WHERE encounter_id = ?", 
                       (json.dumps(history, ensure_ascii=False), diag_summary, encounter_id))
        conn.commit()

        return {"history": history, "latest_action": new_action}

@app.delete("/api/encounters/{encounter_id}/chat/delete")
def delete_chat_message(encounter_id: str, req: DeleteChatRequest):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT history_json FROM encounters WHERE encounter_id = ?", (encounter_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="진료 세션 없음")
        
        history = json.loads(row[0])
        if req.index < 0 or req.index >= len(history):
            raise HTTPException(status_code=400, detail="유효하지 않은 메시지 번호")

        if history[req.index]["role"] == "user":
            if req.index + 1 < len(history) and history[req.index + 1]["role"] == "model":
                del history[req.index:req.index + 2]
            else:
                del history[req.index]
        else:
            del history[req.index]

        cursor.execute("UPDATE encounters SET history_json = ? WHERE encounter_id = ?", 
                       (json.dumps(history, ensure_ascii=False), encounter_id))
        conn.commit()

        return {"history": history}

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>의료 진단 AI System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif; }
        body { background-color: #0f172a; display: flex; justify-content: center; align-items: center; height: 100vh; overflow: hidden; }
        .main-container { width: 100%; max-width: 1080px; height: 95vh; background: #ffffff; border-radius: 14px; display: flex; overflow: hidden; box-shadow: 0 16px 40px rgba(0,0,0,0.5); }
        
        .sidebar { width: 340px; background-color: #f8fafc; border-right: 1px solid #e2e8f0; display: flex; flex-direction: column; }
        .patient-card { padding: 18px 16px; background-color: #f1f5f9; border-bottom: 1px solid #cbd5e1; }
        .patient-card .p-id { font-size: 11px; font-weight: 800; color: #2563eb; text-transform: uppercase; margin-bottom: 2px; }
        .patient-card .p-name { font-size: 17px; font-weight: 800; color: #0f172a; }
        .patient-card .p-meta { font-size: 12px; color: #64748b; margin-top: 3px; }
        
        .enc-header { padding: 14px 16px 8px; display: flex; justify-content: space-between; align-items: center; }
        .enc-header h3 { font-size: 13px; font-weight: 700; color: #334155; }
        .btn-new-enc { background-color: #2563eb; color: white; border: none; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 11.5px; cursor: pointer; }
        
        .history-list { flex: 1; overflow-y: auto; padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; }
        .history-item { padding: 12px 14px; border-radius: 8px; background: #ffffff; border: 1px solid #e2e8f0; cursor: pointer; transition: 0.15s; }
        .history-item:hover { border-color: #2563eb; background-color: #eff6ff; }
        .history-item.active { border-color: #2563eb; background-color: #eff6ff; border-left: 4px solid #2563eb; }
        .history-item .seq { font-size: 11px; font-weight: 800; color: #2563eb; margin-bottom: 2px; }
        .history-item .complaint { font-size: 13px; font-weight: 700; color: #1e293b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .history-item .diag-badge { display: inline-block; font-size: 11px; color: #1e40af; background: #dbeafe; padding: 2px 7px; border-radius: 4px; margin-top: 4px; }
        .history-item .date { font-size: 11px; color: #94a3b8; margin-top: 4px; }

        .chat-section { flex: 1; display: flex; flex-direction: column; background-color: #f8fafc; }
        .chat-header { background-color: #ffffff; padding: 14px 20px; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; }
        .chat-header h1 { font-size: 15px; font-weight: 800; color: #0f172a; }
        .btn-switch-user { font-size: 12px; background: none; border: 1px solid #cbd5e1; padding: 4px 10px; border-radius: 6px; cursor: pointer; color: #64748b; }

        .chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        .msg-row { display: flex; align-items: flex-start; gap: 10px; position: relative; }
        .msg-row.user { flex-direction: row-reverse; }
        .avatar { width: 38px; height: 38px; border-radius: 50%; background-color: #2563eb; color: white; display: flex; align-items: center; justify-content: center; font-size: 17px; flex-shrink: 0; }
        .msg-content { display: flex; flex-direction: column; max-width: 80%; position: relative; }
        
        .bubble { padding: 12px 16px; border-radius: 12px; font-size: 13.5px; line-height: 1.6; word-break: break-word; white-space: pre-wrap; }
        .msg-row.ai .bubble { background-color: #ffffff; color: #0f172a; border: 1px solid #e2e8f0; border-top-left-radius: 2px; }
        .msg-row.user .bubble { background-color: #2563eb; color: #ffffff; border-top-right-radius: 2px; }

        .msg-row.user:hover .msg-actions { display: flex; }
        .msg-actions { display: none; gap: 4px; margin-top: 4px; justify-content: flex-end; }
        .btn-action { background: #e2e8f0; border: none; border-radius: 4px; padding: 2px 6px; font-size: 10.5px; color: #475569; cursor: pointer; }
        .btn-action:hover { background: #cbd5e1; color: #0f172a; }

        .dept-tag { display: inline-flex; align-items: center; font-size: 11px; font-weight: 800; color: #0369a1; background: #e0f2fe; padding: 4px 8px; border-radius: 6px; margin-top: 6px; width: fit-content; }
        
        .diag-card { margin-top: 10px; background-color: #ffffff; border: 1.5px solid #2563eb; border-radius: 10px; padding: 14px; box-shadow: 0 4px 14px rgba(37,99,235,0.12); }
        .diag-card-title { font-size: 13px; font-weight: 800; color: #1e40af; border-bottom: 1px solid #f1f5f9; padding-bottom: 6px; margin-bottom: 8px; display: flex; justify-content: space-between; }
        .diag-primary { font-size: 15px; font-weight: 800; color: #b91c1c; margin-bottom: 6px; }
        .diag-diff { font-size: 12px; color: #475569; margin-bottom: 6px; }
        .diag-reasoning { font-size: 12px; color: #64748b; background: #f8fafc; padding: 8px; border-radius: 6px; line-height: 1.4; }

        .rationale-tag { font-size: 11px; color: #475569; background-color: #f1f5f9; padding: 6px 10px; border-radius: 6px; margin-top: 6px; border-left: 3px solid #2563eb; }
        .chips-container { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
        .chip { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 16px; padding: 6px 12px; font-size: 12px; cursor: pointer; color: #334155; }
        .chip:hover { background-color: #eff6ff; border-color: #2563eb; color: #1d4ed8; }

        /* Gemini 스타일 자동 높이 확장 입력창 */
        .input-bar { background-color: #ffffff; padding: 12px 18px; border-top: 1px solid #e2e8f0; display: flex; gap: 10px; align-items: flex-end; }
        .input-bar textarea { 
            flex: 1; 
            border: 1px solid #cbd5e1; 
            padding: 11px 16px; 
            border-radius: 18px; 
            font-size: 14px; 
            outline: none; 
            resize: none; 
            line-height: 1.5; 
            height: 44px; 
            max-height: 160px; 
            overflow-y: hidden;
            box-sizing: border-box;
            background: #ffffff;
        }
        .input-bar textarea:focus { border-color: #2563eb; }
        .input-bar button { 
            background-color: #2563eb; 
            color: white; 
            border: none; 
            border-radius: 20px; 
            padding: 11px 20px; 
            font-weight: 700; 
            cursor: pointer; 
            height: 44px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; z-index: 100; }
        .modal-content { background: white; width: 92%; max-width: 400px; padding: 24px; border-radius: 12px; }
        .form-group { margin-bottom: 12px; }
        .form-group label { display: block; font-size: 12px; font-weight: 700; color: #475569; margin-bottom: 4px; }
        .form-group input, .form-group select { width: 100%; padding: 9px 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13.5px; }
        .btn-submit { width: 100%; background: #2563eb; color: white; border: none; padding: 12px; border-radius: 6px; font-weight: 700; cursor: pointer; margin-top: 10px; font-size: 14px; }
        /* 📱 모바일/스마트폰 전용 완벽 레이아웃 최적화 */
        @media (max-width: 768px) {
            html, body {
                height: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
                background-color: #f8fafc !important;
            }
            .main-container {
                width: 100% !important;
                height: 100dvh !important;
                max-width: 100% !important;
                border-radius: 0 !important;
                box-shadow: none !important;
                display: flex !important;
                flex-direction: column !important;
            }
            /* 상단 사이드바를 아주 슬림한 환자 요약 탭으로 압축 */
            .sidebar {
                width: 100% !important;
                height: auto !important;
                max-height: none !important;
                flex-shrink: 0 !important;
                border-right: none !important;
                border-bottom: 1px solid #cbd5e1 !important;
                background: #f1f5f9 !important;
            }
            .patient-card {
                padding: 8px 12px !important;
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
                border-bottom: none !important;
            }
            .patient-card .p-id { font-size: 10px !important; margin: 0 !important; }
            .patient-card .p-name { font-size: 14px !important; }
            .patient-card .p-meta { font-size: 11px !important; margin: 0 !important; }

            /* 모바일 진료 이력 토글 서랍 */
            .enc-header {
                display: flex !important;
                padding: 6px 12px !important;
                background: #e2e8f0 !important;
                cursor: pointer;
                font-size: 11.5px !important;
                font-weight: 700 !important;
                color: #334155 !important;
                justify-content: space-between !important;
                align-items: center !important;
            }
            .enc-header::after {
                content: ' ▾ 이력 열기';
                font-size: 10px;
                color: #2563eb;
            }
            .enc-header.open::after {
                content: ' ▴ 접기';
            }
            .history-list {
                display: none;
                max-height: 140px !important;
                overflow-y: auto !important;
                background: #ffffff !important;
                padding: 6px 10px !important;
                gap: 6px !important;
                border-bottom: 1px solid #cbd5e1 !important;
            }
            .history-list.open {
                display: flex !important;
                flex-direction: column !important;
            }
            .history-item {
                padding: 6px 8px !important;
            }
            .history-item .h-title {
                font-size: 11.5px !important;
            }
            .history-item .h-meta {
                font-size: 10px !important;
            }

            /* 채팅 섹션이 스마트폰 전체 화면 차지 */
            .chat-section {
                flex: 1 !important;
                height: auto !important;
                display: flex !important;
                flex-direction: column !important;
                overflow: hidden !important;
            }
            .chat-header {
                padding: 8px 12px !important;
            }
            .chat-header h1 {
                font-size: 12.5px !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                max-width: 70% !important;
            }
            .btn-switch-user {
                padding: 3px 8px !important;
                font-size: 11px !important;
            }

            /* 대화 스크롤 영역 */
            .chat-container {
                flex: 1 !important;
                padding: 10px !important;
                gap: 10px !important;
                overflow-y: auto !important;
            }
            .msg-content {
                max-width: 90% !important;
            }
            .avatar {
                width: 30px !important;
                height: 30px !important;
                font-size: 14px !important;
            }
            .bubble {
                padding: 8px 12px !important;
                font-size: 13px !important;
                line-height: 1.45 !important;
            }

            /* 정밀 진단서 카드 크기 맞춤 */
            .diag-card {
                padding: 10px !important;
                margin-top: 6px !important;
            }
            .diag-primary {
                font-size: 13.5px !important;
            }
            .diag-reasoning {
                font-size: 11.5px !important;
                padding: 6px !important;
            }

            /* 하단 입력창 고정 및 짤림 방지 */
            .input-bar {
                padding: 8px 10px !important;
                gap: 6px !important;
                background: #ffffff !important;
                border-top: 1px solid #e2e8f0 !important;
            }
            .input-bar textarea {
                height: 40px !important;
                font-size: 13.5px !important;
                padding: 8px 12px !important;
            }
            .input-bar button {
                height: 40px !important;
                padding: 0 14px !important;
                font-size: 13px !important;
            }
            .modal-content {
                width: 90% !important;
                padding: 16px !important;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="sidebar">
            <div class="patient-card">
                <div class="p-id" id="displayPatientId">PATIENT NO. ------</div>
                <div class="p-name" id="displayPatientName">환자 정보 확인 중</div>
                <div class="p-meta" id="displayPatientMeta">--세 (--) | 생년월일: ----.--.--</div>
            </div>
            <div class="enc-header">
                <h3>진료 차수 이력</h3>
                <button class="btn-new-enc" onclick="openNewEncounterModal()">+ 새 진료 접수</button>
            </div>
            <div class="history-list" id="encountersList"></div>
        </div>

        <div class="chat-section">
            <div class="chat-header">
                <h1 id="encounterTitle">🩺 전문 임상 진단 엔진</h1>
                <button class="btn-switch-user" onclick="openLoginModal()">환자 변경 / 조회</button>
            </div>
            <div class="chat-container" id="chatContainer"></div>
            <div class="input-bar">
                <textarea 
                    id="userInput" 
                    placeholder="증상 또는 상태를 구체적으로 말씀해 주세요... (Enter: 전송, Shift+Enter: 줄바꿈)" 
                    rows="1"
                    oninput="autoResizeTextarea(this)" 
                    onkeydown="handleTextareaKeydown(event)"
                ></textarea>
                <button type="button" onclick="sendMessage()">전송</button>
            </div>
        </div>
    </div>

    <!-- 환자 등록 모달 -->
    <div class="modal" id="loginModal" style="display:none;">
        <div class="modal-content">
            <h2 style="font-size:17px; margin-bottom:14px; color:#0f172a;">🏥 환자 등록번호 조회 및 본인 확인</h2>
            <div class="form-group">
                <label>환자등록번호</label>
                <input type="text" id="mPatientId" placeholder="ex) PT-2026-001" />
            </div>
            <div class="form-group">
                <label>환자 성함</label>
                <input type="text" id="mPatientName" placeholder="ex) 홍길동" />
            </div>
            <div class="form-group">
                <label>생년월일 (8자리)</label>
                <input type="text" id="mBirthDate" placeholder="ex) 2000.01.01" />
            </div>
            <div class="form-group">
                <label>성별</label>
                <select id="mSex">
                    <option value="남성">남성</option>
                    <option value="여성">여성</option>
                </select>
            </div>
            <button class="btn-submit" onclick="loginPatient()">진료 기록 조회 및 접속</button>
        </div>
    </div>

    <!-- 새 진료 차수 접수 모달 -->
    <div class="modal" id="newEncModal" style="display:none;">
        <div class="modal-content">
            <h2 style="font-size:16px; margin-bottom:12px; color:#0f172a;">진료 접수</h2>
            <div class="form-group">
                <label>불편한 증상</label>
                <input type="text" id="mComplaint" placeholder="ex) 머리가 어지러워요." />
            </div>
            <button class="btn-submit" id="btnStartEnc" onclick="startNewEncounter()">진료 시작하기</button>
            <button type="button" style="width:100%; border:none; background:none; margin-top:8px; font-size:12px; color:#888; cursor:pointer;" onclick="document.getElementById('newEncModal').style.display='none'">취소</button>
        </div>
    </div>

    <script>
        let currentPatient = null;
        let currentEncounterId = null;

        window.onload = function() {
            openLoginModal();
            const encH = document.querySelector('.enc-header');
            if (encH) {
                encH.addEventListener('click', function() {
                    this.classList.toggle('open');
                    const hList = document.querySelector('.history-list');
                    if (hList) hList.classList.toggle('open');
                });
            }
        };

        function openLoginModal() {
            document.getElementById('loginModal').style.display = 'flex';
        }

        async function loginPatient() {
            const pid = document.getElementById('mPatientId').value.trim();
            const pname = document.getElementById('mPatientName').value.trim();
            const pbirth = document.getElementById('mBirthDate').value.trim();
            const psex = document.getElementById('mSex').value;

            if (!pid || !pname) {
                alert('환자번호와 성함을 입력해 주세요.');
                return;
            }

            const res = await fetch('/api/patients/auth', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    patient_id: pid,
                    patient_name: pname,
                    birth_date: pbirth || '2000.01.01',
                    biological_sex: psex
                })
            });

            const data = await res.json();
            currentPatient = data.patient;
            document.getElementById('loginModal').style.display = 'none';

            document.getElementById('displayPatientId').innerText = `PATIENT NO. ${currentPatient.patient_id}`;
            document.getElementById('displayPatientName').innerText = `${currentPatient.patient_name} 님`;
            document.getElementById('displayPatientMeta').innerText = `${currentPatient.biological_sex} | 생년월일: ${currentPatient.birth_date}`;

            renderEncountersList(data.encounters);

            if (data.encounters.length > 0) {
                selectEncounter(data.encounters[0].encounter_id);
            } else {
                openNewEncounterModal();
            }
        }

        function renderEncountersList(encounters) {
            const listContainer = document.getElementById('encountersList');
            listContainer.innerHTML = '';
            encounters.forEach(enc => {
                const item = document.createElement('div');
                item.className = 'history-item' + (enc.encounter_id === currentEncounterId ? ' active' : '');
                item.onclick = () => selectEncounter(enc.encounter_id);
                item.innerHTML = `
                    <div class="seq">제 ${enc.encounter_seq}차 진료</div>
                    <div class="complaint">${enc.chief_complaint}</div>
                    <div><span class="diag-badge">${enc.diagnosis_summary || '문진 진행 중'}</span></div>
                    <div class="date">${enc.created_at}</div>
                `;
                listContainer.appendChild(item);
            });
        }

        async function selectEncounter(encounterId) {
            currentEncounterId = encounterId;
            const res = await fetch(`/api/encounters/${encounterId}`);
            const data = await res.json();

            document.getElementById('encounterTitle').innerText = `제 ${data.encounter_seq}차 진료실: ${data.chief_complaint}`;
            document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));

            renderChatHistory(data.history);
        }

        function renderChatHistory(history) {
            const container = document.getElementById('chatContainer');
            container.innerHTML = '';
            history.forEach((turn, idx) => {
                if (turn.role === 'user') {
                    addUserMessage(turn.content, idx);
                } else {
                    addAiMessage(turn.action || { message: turn.content, rationale: '진료 기록 복원' });
                }
            });
        }

        function openNewEncounterModal() {
            document.getElementById('newEncModal').style.display = 'flex';
        }

        async function startNewEncounter() {
            const complaint = document.getElementById('mComplaint').value.trim();
            if (!complaint) return;

            const btn = document.getElementById('btnStartEnc');
            btn.innerText = 'AI 진료 준비 중...';
            btn.disabled = true;

            const res = await fetch('/api/encounters/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    patient_id: currentPatient.patient_id,
                    chief_complaint: complaint
                })
            });

            const data = await res.json();
            document.getElementById('newEncModal').style.display = 'none';
            btn.innerText = '진료 시작하기';
            btn.disabled = false;

            const authRes = await fetch('/api/patients/auth', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(currentPatient)
            });
            const authData = await authRes.json();
            renderEncountersList(authData.encounters);
            selectEncounter(data.encounter_id);
        }

        function addAiMessage(action) {
            const container = document.getElementById('chatContainer');
            const row = document.createElement('div');
            row.className = 'msg-row ai';

            let deptHtml = action.recommended_department ? `<div class="dept-tag">🏥 추천 전문과: ${action.recommended_department}</div>` : '';

            let diagHtml = '';
            if (action.diagnosis_report && action.diagnosis_report.primary_diagnosis) {
                const d = action.diagnosis_report;
                diagHtml = `
                    <div class="diag-card">
                        <div class="diag-card-title">
                            <span>📋 정밀 임상 진단서</span>
                            <span>확신도: ${d.confidence_score || 75}%</span>
                        </div>
                        <div class="diag-primary">주진단: ${d.primary_diagnosis}</div>
                        ${d.differential_diagnoses && d.differential_diagnoses.length ? `<div class="diag-diff">감별 진단군: ${d.differential_diagnoses.join(', ')}</div>` : ''}
                        <div class="diag-reasoning">${d.clinical_reasoning || ''}</div>
                    </div>
                `;
            }

            let chipsHtml = action.chips && action.chips.length > 0 
                ? '<div class="chips-container">' + action.chips.map(c => `<button type="button" class="chip" onclick="sendMessage('${c.replace(/'/g, "\\\\'")}')">${c}</button>`).join('') + '</div>' 
                : '';

            row.innerHTML = `
                <div class="avatar">👨‍⚕️</div>
                <div class="msg-content">
                    <div class="bubble">${action.message}</div>
                    ${deptHtml}
                    ${diagHtml}
                    <div class="rationale-tag">💡 임상 판단 근거: ${action.rationale}</div>
                    ${chipsHtml}
                </div>
            `;
            container.appendChild(row);
            container.scrollTop = container.scrollHeight;
        }

        function addUserMessage(text, idx) {
            const container = document.getElementById('chatContainer');
            const row = document.createElement('div');
            row.className = 'msg-row user';

            let actionButtons = idx !== undefined ? `
                <div class="msg-actions">
                    <button class="btn-action" onclick="editMessage(${idx}, '${text.replace(/'/g, "\\\\'")}')">수정</button>
                    <button class="btn-action" onclick="deleteMessage(${idx})">삭제</button>
                </div>
            ` : '';

            row.innerHTML = `
                <div class="msg-content">
                    <div class="bubble">${text}</div>
                    ${actionButtons}
                </div>
            `;
            container.appendChild(row);
            container.scrollTop = container.scrollHeight;
        }

        async function editMessage(idx, oldText) {
            const newText = prompt('메시지를 수정하세요:', oldText);
            if (!newText || newText.trim() === oldText) return;

            const res = await fetch(`/api/encounters/${currentEncounterId}/chat/edit`, {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ index: idx, new_text: newText.trim() })
            });

            if (res.ok) {
                const data = await res.json();
                renderChatHistory(data.history);
            } else {
                alert('메시지 수정에 실패했습니다.');
            }
        }

        async function deleteMessage(idx) {
            if (!confirm('이 발언과 관련 답변을 삭제하시겠습니까?')) return;

            const res = await fetch(`/api/encounters/${currentEncounterId}/chat/delete`, {
                method: 'DELETE',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ index: idx })
            });

            if (res.ok) {
                const data = await res.json();
                renderChatHistory(data.history);
            } else {
                alert('메시지 삭제에 실패했습니다.');
            }
        }

        // Gemini 스타일 자동 높이 조절 함수
        function autoResizeTextarea(textarea) {
            textarea.style.height = 'auto';
            const newHeight = Math.min(textarea.scrollHeight, 160);
            textarea.style.height = (newHeight < 44 ? 44 : newHeight) + 'px';
            textarea.style.overflowY = textarea.scrollHeight > 160 ? 'auto' : 'hidden';
        }

        function handleTextareaKeydown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        async function sendMessage(presetText) {
            const textarea = document.getElementById('userInput');
            const text = presetText || textarea.value.trim();
            if (!text || !currentEncounterId) return;

            textarea.value = '';
            textarea.style.height = '44px';
            textarea.style.overflowY = 'hidden';

            try {
                const res = await fetch(`/api/encounters/${currentEncounterId}/respond`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ answer: text })
                });
                const nextAction = await res.json();
                
                // 1. 현재 진료실 대화창 새로고침
                const encRes = await fetch(`/api/encounters/${currentEncounterId}`);
                const encData = await encRes.json();
                renderChatHistory(encData.history);

                // 2. 왼쪽 사이드바 목록도 최신 진단명/확신도로 즉시 동기화
                if (currentPatient) {
                    const pRes = await fetch('/api/patients/auth', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(currentPatient)
                    });
                    const pData = await pRes.json();
                    renderEncountersList(pData.encounters);
                }
            } catch (err) {
                alert('진료 처리 중 통신 오류가 발생했습니다.');
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run("agent_server:app", host="127.0.0.1", port=8000, reload=True)
