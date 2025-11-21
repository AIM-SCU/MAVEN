from __future__ import annotations
import random
from typing import List, Any
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import aiohttp


class LLMRouter:
    """Router that randomly distributes requests across multiple LLM clients."""

    def __init__(self, ports, model="ollama/llama3.1:70b-instruct-q8_0", **kw):
        self.ports = ports
        self.initial_kwargs = kw.copy()
        self.model = model
        self.current_seed = self.initial_kwargs.get('seed', 42)
        self.current_temperature = self.initial_kwargs.get('temperature', 0.4)
        self.clients = self._create_clients_with_kwargs(self.initial_kwargs)
        print("Router ready on ports", ports)

    def _create_clients_with_kwargs(self, kwargs_for_clients: dict) -> List[OpenAIChatCompletionClient]:
        """Helper to create clients with specific kwargs."""
        return [
            OpenAIChatCompletionClient(
                model=self.model,
                api_key="NotRequiredSinceWeAreLocal",
                base_url=f"http://127.0.0.1:{p}",
                model_capabilities={"json_output": True, "vision": False, "function_calling": False},
                **kwargs_for_clients,
            )
            for p in self.ports
        ]

    async def health_check(self) -> List[bool]:
        """Check health of all endpoints."""
        results = []
        for i, port in enumerate(self.ports):
            try:
                async with aiohttp.ClientSession() as session:
                    # Use litellm health endpoint
                    async with session.get(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                        healthy = response.status == 200
                        results.append(healthy)
                        print(f"Port {port}: {'✓' if healthy else '✗'}")
            except Exception as e:
                print(f"Port {port}: ✗ ({e})")
                results.append(False)
        return results

    def get_random_client(self) -> OpenAIChatCompletionClient:
        """Get a randomly selected client."""
        return random.choice(self.clients)

    async def create(self, *args, **kwargs) -> Any:
        """Route request to a random client."""
        client = self.get_random_client()
        return await client.create(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate any other method calls to a random client."""
        def method(*args, **kwargs):
            client = self.get_random_client()
            attr = getattr(client, name)
            if callable(attr):
                return attr(*args, **kwargs)
            return attr
        return method

    def change_seed(self):
        """Change the seed and recreate all clients with the new seed."""
        import random as py_random
        new_seed = py_random.randint(1, 10000)
        while new_seed == self.current_seed:
            new_seed = py_random.randint(1, 10000)

        self.current_seed = new_seed
        print(f"Changing seed to {self.current_seed}")

        new_kwargs = self.initial_kwargs.copy()
        new_kwargs['seed'] = self.current_seed

        self.clients = self._create_clients_with_kwargs(new_kwargs)

    def reset_seed(self):
        """Resets the seed to the initial value from initialization."""
        initial_seed = self.initial_kwargs.get('seed', 42)
        if self.current_seed != initial_seed:
            print(f"Resetting seed to initial value: {initial_seed}")
            self.current_seed = initial_seed
            self.clients = self._create_clients_with_kwargs(self.initial_kwargs)

    def change_temperature(self, new_temperature: float):
        """Change the temperature and recreate all clients with the new temperature."""
        self.current_temperature = new_temperature
        print(f"Changing temperature to {self.current_temperature}")

        new_kwargs = self.initial_kwargs.copy()
        new_kwargs['temperature'] = self.current_temperature
        new_kwargs['seed'] = self.current_seed  # Keep current seed

        self.clients = self._create_clients_with_kwargs(new_kwargs)

    def reset_seed_to_42(self):
        """Reset seed to 42 and recreate all clients."""
        self.current_seed = 42
        print(f"Resetting seed to 42")

        new_kwargs = self.initial_kwargs.copy()
        new_kwargs['seed'] = self.current_seed
        new_kwargs['temperature'] = self.current_temperature  # Keep current temperature

        self.clients = self._create_clients_with_kwargs(new_kwargs)
