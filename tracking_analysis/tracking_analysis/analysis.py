import os
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter

import numpy as np
import pandas as pd

from tracking_analysis.utils import read_in_json


MATCH_DATA = "match_data.json"
STRUCTURED_DATA = "structured_data.json"
HOME_TEAM = "home_team"
AWAY_TEAM = "away_team"
PITCH_LENGTH = "pitch_length"
PITCH_WIDTH = "pitch_width"
TRACKABLE_OBJECT = "trackable_object"
TRACK_ID = "track_id"
BALL = "ball"
REFEREE = "referee"
REFEREES = "referees"
ALL_PLAYERS = "players"
GROUP = "group"

LAST_NAME = "last_name"
TEAM_ID = "team_id"
NUMBER = "number"
PLAYER_ROLE = "player_role"

FRAME = "frame"
POSSESSION = "possession"
PERIOD = "period"
TIME = "time"


class IdMismatchException(Exception):
    pass


class Player(object):
    def __init__(self,
                 player_json: Dict[str, Any]
                 ) -> None:
        self.player_json = player_json
        self.last_name = self._get(LAST_NAME)
        self.player_id = self._get(TRACKABLE_OBJECT)
        self.team_id = self._get(TEAM_ID)
        self.number = self._get(NUMBER)
        self.position, self.position_id = self._get_position_info(self.player_json.get(PLAYER_ROLE, {}))

    def __repr__(self):
        return f"{self.last_name} ({self.player_id})"

    def _get(self, key: str):
        return self.player_json.get(key, None)

    @staticmethod
    def _get_position_info(player_role_dict: Dict[str, Any]) -> Tuple[str, int]:
        position = player_role_dict.get("acronym", None)
        id_ = player_role_dict.get("id", None)
        return position, id_


class Coordinate(object):
    def __init__(self, x: float, y: float, z: Optional[float]):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"({self.x, self.y})" if not self.z else f"({self.x, self.y, self.z})"


class Location(object):
    def __init__(self,
                 location_json: Dict[str, Any],
                 period: int
                 ):
        self.location_json = location_json
        self.x = self._get("x")
        self.y = self._get("y")
        self.normalized_y = self._flip_y(self.y) if period == 2 else self.y     # normalize attacking half of pitch
        self.z = self._get("z")
        self.location_id = self._get(TRACKABLE_OBJECT)
        self.track_id = self._get(TRACK_ID)

    def _get(self, key: str):
        return self.location_json.get(key, None)

    @staticmethod
    def _flip_y(y: float):
        return -y


class Moment(object):
    def __init__(self,
                 moment_json: Dict[str, Any]
                 ) -> None:
        self.moment_json = moment_json
        self.frame = self._get(FRAME)
        self.time = self._get(TIME)
        self.half = self._get(PERIOD)
        self.locations = [Location(l, self.half) for l in self.moment_json.get("data", [])]
        self.possession_id, self.possession_group = self._get_possession()

    def __repr__(self):
        return f"Moment at {self.time}; possession={self.possession_group}"

    def _get(self, key: str):
        return self.moment_json.get(key, None)

    def _get_possession(self):
        possession_dict = self._get(POSSESSION)
        return possession_dict.get(TRACKABLE_OBJECT), possession_dict.get(GROUP)


class Match(object):
    def __init__(self,
                 match_json: Dict[str, Any],
                 structured_json: List[Dict[str, Any]]
                 ) -> None:
        self.match_json = match_json
        self.structured_json = structured_json
        self.home_team_name, self.home_team_id = self._get_team_info(HOME_TEAM)
        self.away_team_name, self.away_team_id = self._get_team_info(AWAY_TEAM)
        self.team_to_id = {self.home_team_name: self.home_team_id, self.away_team_name: self.away_team_id}
        self.id_to_team = {v: k for k, v in self.team_to_id.items()}
        self.pitch_size = self._get_pitch_size()
        self.ball_id = self._get_trackable_object(BALL)
        self.referee_id = self._get_referee(REFEREES)
        self._get_all_players()
        self._build_lookups()
        self.moments = [Moment(m) for m in self.structured_json]

    def __repr__(self):
        return f"{self.home_team_name} v. {self.away_team_name}"

    def _get_team_info(self, key: str):
        acro = self.match_json.get(key, {}).get("acronym")
        id_ = self.match_json.get(key, {}).get("id")
        return acro, id_

    def _get_pitch_size(self):
        length = self.match_json.get(PITCH_LENGTH, None)
        width = self.match_json.get(PITCH_WIDTH, None)
        return length, width

    def _get_trackable_object(self, key: str):
        return self.match_json.get(key, {}).get(TRACKABLE_OBJECT, None)

    def _get_referee(self, key: str):
        return self.match_json.get(key, [])[0].get(TRACKABLE_OBJECT, None)

    @staticmethod
    def _get_player(player_json: Dict[str, Any]) -> Player:
        return Player(player_json)

    def _get_all_players(self):
        self.home_team_players = []
        self.away_team_players = []
        for p_dict in self.match_json.get(ALL_PLAYERS, []):
            p = self._get_player(p_dict)
            if p.team_id == self.home_team_id:
                self.home_team_players.append(p)
            elif p.team_id == self.away_team_id:
                self.away_team_players.append(p)
            else:
                raise IdMismatchException(
                    f"Player's team id, {p.team_id} not in {set([self.home_team_id, self.away_team_id])}"
                )
        self.all_players = self.home_team_players + self.away_team_players

    def _build_lookups(self):
        self.id_to_name = {p.player_id: p.last_name for p in self.home_team_players + self.away_team_players}
        self.id_to_name[self.ball_id] = BALL
        self.id_to_name[self.referee_id] = REFEREE
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}


class StringOfPossession(object):
    def __init__(self,
                 string_of_possession: List[Moment],
                 n2i: Dict[str, int],
                 i2n: Dict[int, str]
                 ):
        self.team = string_of_possession[0].possession_group
        self.moments = string_of_possession
        self.n2i = n2i
        self.i2n = i2n

    def __repr__(self):
        string = "<s>"
        previous_possession = None
        for m in self.string_of_possession:
            if m.possession_id and previous_possession != m.possession_id:
                string += f" -> {self.i2n[m.possession_id]}"
                previous_possession = m.possession_id
        return string


class Analysis(object):
    def __init__(self,
                 path_to_data_dir: str
                 ) -> None:
        self.match_data, self.structured_data = self._import_match_data(path_to_data_dir)
        self.match = Match(self.match_data, self.structured_data)

    def __repr__(self):
        return f"Analysis of {self.match}"

    @staticmethod
    def _import_match_data(path_to_data_dir):
        match_data = read_in_json(os.path.join(path_to_data_dir, MATCH_DATA))
        structured_data = read_in_json(os.path.join(path_to_data_dir, STRUCTURED_DATA))
        return match_data, structured_data

    def get_team_possession_stats(self) -> Dict[str, float]:
        home_count = 0
        away_count = 0
        for m in self.match.moments:
            if m.possession_group == "home team":
                home_count += 1
            elif m.possession_group == "away team":
                away_count += 1
        total_count = home_count + away_count
        home_percentage = np.round(home_count / total_count, 3)
        away_percentage = np.round(away_count / total_count, 3)
        return {
            self.match.home_team_name: home_percentage,
            self.match.away_team_name: away_percentage
        }

    def get_player_possession_stats(self) -> Counter:
        raw_counts = Counter(
            [self.match.id_to_name.get(m.possession_id, None) for m in self.match.moments if m.possession_id]
        )
        return Counter(
            {n: np.round(raw_counts[n] / sum(raw_counts.values()), 2) for n, _ in raw_counts.most_common()}
        )

    @staticmethod
    def _build_possession_strings(list_of_moments: List[Moment]) -> List[List[Moment]]:
        # possession strings
        possession_strings = []
        previous_possession = None
        possession_buffer = []
        for m in sorted(list_of_moments, key=lambda x: x.frame):
            if previous_possession != m.possession_group:
                # create a new string
                possession_strings.append(possession_buffer)
                possession_buffer = [m]
                previous_possession = m.possession_group
            else:
                # continue existing possession string
                possession_buffer.append(m)
                previous_possession = m.possession_group
        # filter out none possession strings
        return [p for p in possession_strings if set([pp.possession_group for pp in p]) != {None}]

    def get_possession_strings(self) -> List[StringOfPossession]:
        possession_strings_raw = self._build_possession_strings(self.match.moments)
        return [StringOfPossession(sop, self.match.name_to_id, self.match.id_to_name) for sop in possession_strings_raw]

    def build_location_dataframe(self):
        location_dict = {
            (m.frame, m.time): {
                location.location_id: Coordinate(location.x, location.y, location.z) for location in m.locations
            } for m in self.match.moments
        }
        return pd.DataFrame.from_dict(location_dict, orient="index")
