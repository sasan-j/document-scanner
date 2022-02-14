import json
import subprocess
import pathlib


class Experiment:
    """
    Class to store results of any experiment
    """

    def __init__(self, name, args, output_dir: pathlib.Path = pathlib.Path("../")):
        self.gitHash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode(
            "utf-8"
        )
        print(self.gitHash)
        if args is not None:
            self.name = name
            self.params = args
            self.results = {}
            self.dir = output_dir

            import datetime

            now = datetime.datetime.now()
            date_part = now.strftime("%Y-%m-%d")
            time_part = now.strftime("%H-%M-%S")
            if (
                experiment_base := output_dir / self.name / date_part
            ).exists() is False:
                experiment_base.mkdir(parents=True)
            ver = 0

            while (experiment_base / f"{time_part}_{ver}").exists():
                ver += 1

            self.path = experiment_base / f"{time_part}_{ver}"
            self.path.mkdir(parents=True)

            self.results["Temp Results"] = [[1, 2, 3, 4], [5, 6, 2, 6]]

    def store_json(self):
        with open(self.path / "JSONDump.txt", "w") as outfile:
            json.dump(json.dumps(self.__dict__), outfile)
