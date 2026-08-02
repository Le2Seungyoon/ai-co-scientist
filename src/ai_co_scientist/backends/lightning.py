"""Lightning AI Studio GPU 제어 — 로컬 절전/세션 종료와 무관하게 원격에서 학습을 돌리기 위한 얇은 래퍼.

detached 실행(`nohup ... &`)이 기본 패턴이다: 로컬이 죽어도 원격 학습은 살아남고,
재접속해 로그/산출물만 회수한다. lightning_sdk import는 함수 안에서 — 이 그룹이
설치되지 않은 환경에서도 모듈 import 자체는 실패하지 않게 한다.
"""
import os

from ai_co_scientist.core.config import load_dotenv

DEFAULT_STUDIO = "cosci-sem-depth"


def _require_env() -> str:
    load_dotenv()
    api_key = os.environ.get("LIGHTNING_API_KEY")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE")
    if not (api_key and teamspace):
        raise RuntimeError(
            "LIGHTNING_API_KEY/LIGHTNING_TEAMSPACE 필요 (.env 참고, README Lightning 활성화)")
    return teamspace


def _username() -> str:
    from lightning_sdk.lightning_cloud.rest_client import LightningClient
    return LightningClient().auth_service_get_user().username


def studio(name: str = DEFAULT_STUDIO):
    """기존 Studio 핸들 (생성은 하지 않음 — 오타로 새 Studio가 생기는 사고 방지)."""
    from lightning_sdk import Studio
    return Studio(name=name, teamspace=_require_env(), user=_username(), create_ok=False)


def ensure_running(st, machine: str | None = None) -> str:
    """Studio를 켜고 (요청 시) 머신 타입을 전환. 현재 상태 문자열을 반환."""
    from lightning_sdk import Machine
    if str(st.status) != "Running":
        st.start()
    if machine and machine not in str(st.machine):
        st.switch_machine(Machine.from_str(machine))
    return f"{st.status} {st.machine}"


def upload(st, local: str, remote: str) -> None:
    st.upload_file(local, remote)


def exec_cmd(st, cmd: str) -> str:
    return st.run(cmd)


def download(st, remote: str, local: str) -> None:
    st.download_file(remote, local)


def stop(st) -> str:
    """CPU로 내린 뒤 정지 — GPU 머신을 켜둔 채 잊어버리는 크레딧 사고 방지."""
    from lightning_sdk import Machine
    st.switch_machine(Machine.CPU)
    st.stop()
    return str(st.status)


def get_credits() -> float:
    from lightning_sdk.lightning_cloud.rest_client import LightningClient
    client = LightningClient()
    teamspace = _require_env()
    projects = client.projects_service_list_memberships()
    project_id = next(p.project_id for p in projects.memberships if p.name == teamspace)
    return float(client.billing_service_get_project_balance(project_id=project_id).balance)
