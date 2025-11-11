# app.py (version améliorée : Auth + Audit + UI améliorée)
import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import datetime, date
import math
import os
import hashlib

# ---------------------------
# Config & Defaults
# ---------------------------
DB_PATH = "priorisation.db"

DEFAULT_ENGINEERS = {
    "civil": 4,
    "electricien": 3,
    "mecanicien": 2,
    "logisticien": 2,
    "chef_de_projet": 1
}

DEFAULT_WEIGHTS = {
    "C1_engineers": 0.25,
    "C2_progress": 0.20,
    "C3_client_avail": 0.10,
    "C4_location": 0.10,
    "C5_delay": 0.10,
    "C6_client_type": 0.10,
    "C9_age_priority": 0.15
}

CLIENT_TYPE_SCORES = {
    "VIP": 1.0,
    "vip": 1.0,
    "militaire": 0.9,
    "public": 0.7,
    "privé": 0.5,
    "prive": 0.5,
    "private": 0.5
}

WILAYA_TO_ZONE = {
    "alger": "Alger",
    "oran": "Oran",
    "constantine": "Constantine",
    "bejaia": "Alger",
    "béjaïa": "Alger",
    "béchar": "Sud",
    "tamanrasset": "Sud"
}

WILAYA_ZONE_SCORES = {
    "Alger": 0.8,
    "Oran": 0.7,
    "Constantine": 0.6,
    "Sud": 0.5,
    "Other": 0.6
}

MAX_DELAY_DAYS = 30
MAX_AGE_DAYS = 365

# ---------------------------
# DB init & helpers
# ---------------------------
def get_conn():
    need_create = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    if need_create:
        init_db(conn)
    return conn

def init_db(conn):
    c = conn.cursor()
    # projects
    c.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    # sites
    c.execute("""
    CREATE TABLE IF NOT EXISTS sites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        site_name TEXT,
        c1_json TEXT,
        c2_gros_oeuvre REAL,
        c2_desinstallation REAL,
        c2_menuiserie REAL,
        c2_electricite REAL,
        c2_clim REAL,
        c3_personnel_client INTEGER,
        c4_wilaya TEXT,
        c5_retard_jours REAL,
        c6_type_client TEXT,
        c9_date_demande TEXT,
        priorite_absolue INTEGER DEFAULT 0,
        created_at TEXT,
        updated_at TEXT,
        FOREIGN KEY(project_id) REFERENCES projects(id)
    )
    """)
    # engineers
    c.execute("""
    CREATE TABLE IF NOT EXISTS engineers (
        specialty TEXT PRIMARY KEY,
        available INTEGER
    )
    """)
    # weights
    c.execute("""
    CREATE TABLE IF NOT EXISTS weights (
        key TEXT PRIMARY KEY,
        value REAL
    )
    """)
    # users (auth)
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TEXT
    )
    """)
    # audit trail
    c.execute("""
    CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT,
        details TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()

    # seed default engineers & weights if empty
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM engineers")
    if cur.fetchone()[0] == 0:
        for k, v in DEFAULT_ENGINEERS.items():
            cur.execute("INSERT INTO engineers (specialty, available) VALUES (?, ?)", (k, int(v)))
    cur.execute("SELECT COUNT(*) FROM weights")
    if cur.fetchone()[0] == 0:
        for k, v in DEFAULT_WEIGHTS.items():
            cur.execute("INSERT INTO weights (key, value) VALUES (?, ?)", (k, float(v)))
    # seed admin user if no users
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        # default admin: username=admin password=admin123 (change après)
        admin_pw = hash_password("admin123", "admin")
        cur.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                    ("admin", admin_pw, "admin", datetime.now().isoformat()))
    conn.commit()

# ---------------------------
# Auth helpers
# ---------------------------
def hash_password(password: str, username: str) -> str:
    """Hash using sha256 of username + password (simple)."""
    s = (username + ":" + password).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def verify_password(password: str, username: str, stored_hash: str) -> bool:
    return hash_password(password, username) == stored_hash

def create_user(conn, username, password, role="user"):
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                    (username, hash_password(password, username), role, datetime.now().isoformat()))
        conn.commit()
        return True, None
    except Exception as e:
        return False, str(e)

# ---------------------------
# Audit logging
# ---------------------------
def log_action(conn, username, action, details=""):
    cur = conn.cursor()
    cur.execute("INSERT INTO audit (username, action, details, timestamp) VALUES (?, ?, ?, ?)",
                (username, action, str(details), datetime.now().isoformat()))
    conn.commit()

# ---------------------------
# Scoring functions (same logic)
# ---------------------------
def parse_c1_requirement_dict(c1_json):
    if not c1_json:
        return {}
    if isinstance(c1_json, dict):
        return c1_json
    try:
        return json.loads(c1_json)
    except Exception:
        text = str(c1_json)
        parts = [p.strip() for p in text.replace("|", ";").replace(",", ";").split(";") if p.strip()]
        req = {}
        for p in parts:
            tokens = p.split()
            try:
                num = int(tokens[0])
                spec = " ".join(tokens[1:]).lower()
            except Exception:
                num = 1
                spec = p.lower()
            spec = spec.replace("é","e").replace("è","e").replace(" ","_")
            req[spec] = req.get(spec, 0) + num
        return req

def score_engineers(required_dict, available_dict):
    if not required_dict:
        return 1.0
    total_required = sum(required_dict.values())
    if total_required == 0:
        return 1.0
    covered = 0.0
    for spec, req_n in required_dict.items():
        spec_key = spec
        avail_n = 0
        for k in available_dict:
            if spec_key in k or k in spec_key or spec_key.replace("_"," ") in k:
                avail_n = available_dict[k]
                break
        if avail_n == 0:
            for k in available_dict:
                if k.replace("_","") in spec_key.replace("_","") or spec_key.replace("_","") in k.replace("_",""):
                    avail_n = available_dict[k]
                    break
        coverage = min(avail_n, req_n) / req_n if req_n > 0 else 1.0
        covered += coverage * req_n
    return covered / total_required

def score_progress(row):
    phases = ["c2_gros_oeuvre","c2_desinstallation","c2_menuiserie","c2_electricite","c2_clim"]
    vals = []
    for p in phases:
        v = row.get(p, None)
        try:
            v = float(v)
        except Exception:
            v = None
        if v is None:
            continue
        vals.append(max(0.0, min(100.0, v)))
    if not vals:
        return 0.5
    avg = sum(vals)/len(vals)
    return 1.0 - (avg/100.0)

def score_client_availability(v):
    try:
        n = float(v)
    except Exception:
        return 0.5
    if n >= 5:
        return 1.0
    if n >= 3:
        return 0.6
    if n <= 0:
        return 0.0
    if n < 3:
        return 0.6 * (n/3.0)
    else:
        return 0.6 + (((n-3.0)/2.0) * 0.4)

def score_location(wilaya):
    if wilaya is None:
        return WILAYA_ZONE_SCORES.get("Other", 0.6)
    w = str(wilaya).strip().lower()
    zone = WILAYA_TO_ZONE.get(w, None)
    if not zone:
        wsimple = w.replace(" ","").replace("é","e")
        for k in WILAYA_TO_ZONE:
            if k.replace(" ","").replace("é","e") == wsimple:
                zone = WILAYA_TO_ZONE[k]
                break
    if not zone:
        return WILAYA_ZONE_SCORES.get("Other", 0.6)
    return WILAYA_ZONE_SCORES.get(zone, WILAYA_ZONE_SCORES.get("Other", 0.6))

def score_delay(days):
    try:
        d = float(days)
    except Exception:
        return 0.0
    if d <= 0:
        return 0.0
    return min(d / MAX_DELAY_DAYS, 1.0)

def score_client_type(t):
    if t is None:
        return 0.5
    s = CLIENT_TYPE_SCORES.get(str(t).strip(), None)
    if s is None:
        s = CLIENT_TYPE_SCORES.get(str(t).strip().lower(), 0.5)
    return s

def score_age_priority(date_str):
    if not date_str:
        return 0.0
    try:
        dt = pd.to_datetime(date_str)
    except Exception:
        return 0.0
    delta = datetime.now() - dt.to_pydatetime()
    days = max(0.0, delta.days)
    return min(days / MAX_AGE_DAYS, 1.0)

# ---------------------------
# Compute ranking
# ---------------------------
def compute_scores_for_project(conn, project_id):
    df_sites = pd.read_sql_query("SELECT * FROM sites WHERE project_id = ?", conn, params=(project_id,))
    eng_df = pd.read_sql_query("SELECT * FROM engineers", conn)
    available = {r['specialty']: int(r['available']) for _, r in eng_df.iterrows()}
    w_df = pd.read_sql_query("SELECT * FROM weights", conn)
    weights = {r['key']: float(r['value']) for _, r in w_df.iterrows()}
    ssum = sum(weights.values()) if weights else 0
    if ssum != 0:
        for k in weights:
            weights[k] = weights[k] / ssum

    rows = []
    for _, r in df_sites.iterrows():
        c1 = parse_c1_requirement_dict(r['c1_json'])
        s_c1 = score_engineers(c1, available)
        s_c2 = score_progress(r)
        s_c3 = score_client_availability(r['c3_personnel_client'])
        s_c4 = score_location(r['c4_wilaya'])
        s_c5 = score_delay(r['c5_retard_jours'])
        s_c6 = score_client_type(r['c6_type_client'])
        s_c9 = score_age_priority(r['c9_date_demande'])
        weighted = (
            s_c1 * weights.get("C1_engineers", 0) +
            s_c2 * weights.get("C2_progress", 0) +
            s_c3 * weights.get("C3_client_avail", 0) +
            s_c4 * weights.get("C4_location", 0) +
            s_c5 * weights.get("C5_delay", 0) +
            s_c6 * weights.get("C6_client_type", 0) +
            s_c9 * weights.get("C9_age_priority", 0)
        )
        rows.append({
            "id": r['id'],
            "site_name": r['site_name'],
            "c1_required": json.dumps(c1, ensure_ascii=False),
            "score_C1_engineers": round(s_c1, 4),
            "score_C2_progress": round(s_c2, 4),
            "score_C3_client_avail": round(s_c3, 4),
            "score_C4_location": round(s_c4, 4),
            "score_C5_delay": round(s_c5, 4),
            "score_C6_client_type": round(s_c6, 4),
            "score_C9_age_priority": round(s_c9, 4),
            "weighted_score": round(weighted, 6),
            "priorite_absolue": bool(r['priorite_absolue']),
            "c3_personnel_client": r['c3_personnel_client'],
            "c4_wilaya": r['c4_wilaya'],
            "c5_retard_jours": r['c5_retard_jours'],
            "c6_type_client": r['c6_type_client'],
            "c9_date_demande": r['c9_date_demande']
        })
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    out_df["_priority_group"] = out_df["priorite_absolue"].apply(lambda x: 0 if x else 1)
    out_df = out_df.sort_values(by=["_priority_group", "weighted_score"], ascending=[True, False]).reset_index(drop=True)
    out_df["rank"] = out_df.index + 1
    return out_df

# ---------------------------
# UI : Streamlit
# ---------------------------
st.set_page_config(page_title="Priorisation Sites - Auth & Audit", layout="wide")
st.title("🔧 Priorisation automatique des sites — with Auth & Audit")

conn = get_conn()
init_db(conn)

# -------- Sidebar : Auth / user session --------
if "user" not in st.session_state:
    st.session_state["user"] = None

st.sidebar.header("Utilisateur")

if st.session_state["user"] is None:
    auth_tab = st.sidebar.radio("Action", ["Se connecter", "S'inscrire", "Info demo"])
    if auth_tab == "Se connecter":
        uname = st.sidebar.text_input("Nom d'utilisateur")
        pwd = st.sidebar.text_input("Mot de passe", type="password")
        if st.sidebar.button("Connexion"):
            cur = conn.cursor()
            cur.execute("SELECT password_hash, role FROM users WHERE username=?", (uname,))
            row = cur.fetchone()
            if row and verify_password(pwd, uname, row[0]):
                st.session_state["user"] = {"username": uname, "role": row[1]}
                log_action(conn, uname, "login", "Connexion réussie")
                st.rerun()
            else:
                st.sidebar.error("Identifiants invalides")
    elif auth_tab == "S'inscrire":
        new_user = st.sidebar.text_input("Nom d'utilisateur (nouveau)")
        new_pwd = st.sidebar.text_input("Mot de passe", type="password")
        if st.sidebar.button("Créer un compte"):
            ok, err = create_user(conn, new_user, new_pwd)
            if ok:
                st.sidebar.success("Utilisateur créé. Connecte-toi.")
                log_action(conn, new_user, "register", "Nouvel utilisateur créé")
            else:
                st.sidebar.error(f"Erreur création utilisateur: {err}")
    else:
        st.sidebar.markdown("**Demo**: un admin par défaut existe -> `admin / admin123`.")
else:
    st.sidebar.markdown(f"Connecté en tant que **{st.session_state['user']['username']}** ({st.session_state['user']['role']})")
    if st.sidebar.button("Se déconnecter"):
        log_action(conn, st.session_state['user']['username'], "logout", "Utilisateur déconnecté")
        st.session_state["user"] = None
        st.experimental_rerun()

# -------- Sidebar : Projects --------
st.sidebar.header("Projets")
projects_df = pd.read_sql_query("SELECT * FROM projects ORDER BY id DESC", conn)
proj_options = ["-- Nouveau projet --"] + [f"{r['id']} - {r['name']}" for _, r in projects_df.iterrows()]
sel_proj = st.sidebar.selectbox("Choisir un projet", proj_options)

# helper: require login
def require_login():
    if st.session_state["user"] is None:
        st.warning("Tu dois être connecté pour effectuer cette action.")
        return False
    return True

# Create project UI
if sel_proj == "-- Nouveau projet --":
    st.sidebar.text_input("Nom du projet (pour créer)", key="new_project_name")
    if st.sidebar.button("Créer projet"):
        if not require_login():
            pass
        else:
            name = st.session_state.get("new_project_name") or f"Projet {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            cur = conn.cursor()
            cur.execute("INSERT INTO projects (name, created_at) VALUES (?, ?)", (name, datetime.now().isoformat()))
            conn.commit()
            log_action(conn, st.session_state["user"]["username"], "create_project", {"project_name": name})
            st.experimental_rerun()
else:
    project_id = int(sel_proj.split(" - ")[0])
    project_name = sel_proj.split(" - ", 1)[1]
    st.sidebar.markdown(f"**Projet:** {project_name} (ID {project_id})")
    if st.sidebar.button("Supprimer projet"):
        if not require_login(): pass
        else:
            cur = conn.cursor()
            cur.execute("DELETE FROM sites WHERE project_id=?", (project_id,))
            cur.execute("DELETE FROM projects WHERE id=?", (project_id,))
            conn.commit()
            log_action(conn, st.session_state["user"]["username"], "delete_project", {"project_id": project_id})
            st.success("Projet supprimé.")
            st.experimental_rerun()

# Main area: project tabs
if sel_proj != "-- Nouveau projet --":
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Ajouter site", "📊 Classement & UI", "⚙️ Paramètres", "📝 Audit (historique)"])

    with tab1:
        st.header("Ajouter / éditer un site")
        with st.form("add_site"):
            site_name = st.text_input("Nom du site")
            col1, col2 = st.columns(2)
            with col1:
                eng_df = pd.read_sql_query("SELECT * FROM engineers", conn)
                specialties = list(eng_df['specialty'].values)
                st.write("Besoin ingénieurs (C1)")
                reqs = {}
                for sp in specialties:
                    val = st.number_input(f"{sp.replace('_',' ')}", min_value=0, value=0, key=f"req_{sp}")
                    reqs[sp] = int(val)
            with col2:
                st.write("Avancement (C2) %")
                gros = st.slider("Gros oeuvre", 0, 100, 0)
                desins = st.slider("Désinstallation", 0, 100, 0)
                men = st.slider("Menuiserie", 0, 100, 0)
                elec = st.slider("Électricité", 0, 100, 0)
                clim = st.slider("Climatisation", 0, 100, 0)

            c3 = st.selectbox("Disponibilité client (C3)", [5,3,0], index=0)
            c4 = st.text_input("Wilaya (C4)", value="")
            c5 = st.number_input("Retard équipement (jours) (C5)", min_value=0, value=0)
            c6 = st.selectbox("Type client (C6)", ["VIP","public","privé","militaire"], index=1)
            c9 = st.date_input("Date de demande (C9)", value=date.today())
            prior = st.checkbox("Priorité absolue")
            submitted = st.form_submit_button("Ajouter le site")
            if submitted:
                if not require_login():
                    st.stop()
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO sites (project_id, site_name, c1_json, c2_gros_oeuvre, c2_desinstallation, c2_menuiserie,
                                       c2_electricite, c2_clim, c3_personnel_client, c4_wilaya, c5_retard_jours,
                                       c6_type_client, c9_date_demande, priorite_absolue, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id,
                    site_name,
                    json.dumps(reqs, ensure_ascii=False),
                    float(gros), float(desins), float(men), float(elec), float(clim),
                    int(c3),
                    c4,
                    float(c5),
                    c6,
                    pd.to_datetime(c9).isoformat(),
                    1 if prior else 0,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                conn.commit()
                log_action(conn, st.session_state["user"]["username"], "create_site", {"project_id": project_id, "site": site_name, "c1": reqs})
                st.success("Site ajouté ✅")
                st.experimental_rerun()

    with tab2:
        st.header("Classement — visualisation améliorée")
        scores_df = compute_scores_for_project(conn, project_id)
        if scores_df.empty:
            st.info("Aucun site. Ajoute-en depuis l'onglet 'Ajouter site'.")
        else:
            # Top 3 cards
            st.subheader("Top 3 sites — Priorité")
            top3 = scores_df.head(3)
            cols = st.columns(3)
            for idx, (_, row) in enumerate(top3.iterrows()):
                c = cols[idx]
                # badge for priority absolute
                badge = "🔥 PRIORITAIRE" if row['priorite_absolue'] else ""
                c.markdown(f"#### {row['rank']}. {row['site_name']}  <small style='color:orange'>{badge}</small>", unsafe_allow_html=True)
                c.metric("Score pondéré", f"{row['weighted_score']:.4f}", delta=None)
                c.write(f"Type client: **{row['c6_type_client']}** • Wilaya: **{row['c4_wilaya']}**")
                # mini breakdown
                c.write(f"- C1: {row['score_C1_engineers']} • C2: {row['score_C2_progress']} • C5: {row['score_C5_delay']}")
            st.markdown("---")

            # Score distribution chart
            st.subheader("Distribution des scores")
            chart_df = scores_df[['site_name','weighted_score']].set_index('site_name').sort_values(by='weighted_score', ascending=True)
            st.bar_chart(chart_df)

            st.markdown("---")
            st.subheader("Tableau complet des scores")
            st.dataframe(scores_df.drop(columns=["_priority_group"]), use_container_width=True)

            # details panel per site
            st.markdown("### Détails par site")
            selected = st.selectbox("Choisir un site pour voir le détail", options=list(scores_df['site_name']))
            sel_row = scores_df[scores_df['site_name']==selected].iloc[0]
            st.json({
                "C1_required": json.loads(sel_row['c1_required']),
                "score_C1_engineers": sel_row['score_C1_engineers'],
                "score_C2_progress": sel_row['score_C2_progress'],
                "score_C3_client_avail": sel_row['score_C3_client_avail'],
                "score_C4_location": sel_row['score_C4_location'],
                "score_C5_delay": sel_row['score_C5_delay'],
                "score_C6_client_type": sel_row['score_C6_client_type'],
                "score_C9_age_priority": sel_row['score_C9_age_priority'],
                "weighted_score": sel_row['weighted_score']
            })

    with tab3:
        st.header("Paramètres — Ingénieurs & Pondérations")
        st.subheader("Disponibilités d'ingénieurs")
        eng_df = pd.read_sql_query("SELECT * FROM engineers", conn)
        with st.form("eng_form"):
            eng_updates = {}
            for _, r in eng_df.iterrows():
                key = r['specialty']
                val = st.number_input(f"{key.replace('_',' ')}", min_value=0, value=int(r['available']), key=f"eng_{key}")
                eng_updates[key] = int(val)
            if st.form_submit_button("Enregistrer disponibilités"):
                if not require_login(): st.stop()
                cur = conn.cursor()
                for k,v in eng_updates.items():
                    cur.execute("UPDATE engineers SET available=? WHERE specialty=?", (v, k))
                conn.commit()
                log_action(conn, st.session_state["user"]["username"], "update_engineers", eng_updates)
                st.success("Disponibilités mises à jour.")
                st.experimental_rerun()

        st.subheader("Pondérations (normalisées)")
        w_df = pd.read_sql_query("SELECT * FROM weights", conn)
        with st.form("weights_form"):
            w_updates = {}
            for _, r in w_df.iterrows():
                k = r['key']
                val = st.number_input(f"{k}", min_value=0.0, value=float(r['value']), format="%.4f", key=f"w_{k}")
                w_updates[k] = float(val)
            if st.form_submit_button("Enregistrer pondérations"):
                if not require_login(): st.stop()
                cur = conn.cursor()
                total = sum(w_updates.values()) if sum(w_updates.values())>0 else 1.0
                for k,v in w_updates.items():
                    cur.execute("UPDATE weights SET value=? WHERE key=?", (v/total, k))
                conn.commit()
                log_action(conn, st.session_state["user"]["username"], "update_weights", w_updates)
                st.success("Pondérations mises à jour.")
                st.experimental_rerun()

    with tab4:
        st.header("Audit trail (historique des actions)")
        # only admin can see all; others see their own actions
        if st.session_state["user"] and st.session_state["user"]["role"]=="admin":
            audit_q = "SELECT * FROM audit ORDER BY id DESC LIMIT 500"
            audit_df = pd.read_sql_query(audit_q, conn)
            st.dataframe(audit_df, use_container_width=True)
            # filter
            usr_filter = st.text_input("Filtrer par utilisateur (laisser vide pour tout)")
            if usr_filter:
                st.dataframe(audit_df[audit_df['username'].str.contains(usr_filter)], use_container_width=True)
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM audit WHERE username=? ORDER BY id DESC LIMIT 500", (st.session_state["user"]["username"],) if st.session_state["user"] else ("",))
            rows = cur.fetchall()
            cols = ["id","username","action","details","timestamp"]
            audit_df = pd.DataFrame(rows, columns=cols)
            st.dataframe(audit_df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("App prototype: Authentification basique, audit trail, UX améliorée — personnalise selon besoins (ex: OAuth, protection des mots de passe renforcée).")
