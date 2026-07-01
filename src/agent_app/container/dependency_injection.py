from dependency_injector import containers
from dependency_injector import providers

from agent_app.agents.response_struct.agent import ResponseStructAgent
from agent_app.agents.sql_builder.agent import SqlBuilderAgent
from agent_app.application.usecase.database_schema.impl import DatabaseSchemaUsecase
from agent_app.infrastructure.repository.database_schema import DatabaseSchemaRepository
from agent_app.infrastructure.repository.db_manager import DatabaseManager
from agent_app.shared.database.connection.qdrant import QdrantDBHelper
from agent_app.shared.embedding.embedding import LocalSentenceEmbeddings
from shared.config.model import Config, EmbeddingConfig, load_config, HuggingFaceConfig


class Container(containers.DeclarativeContainer):
    config = providers.Dependency()
    
    db_manager = providers.Factory(DatabaseManager, database=config.provided.database)
    local_embedding = providers.Factory(LocalSentenceEmbeddings, 
        model_name=(config.provided.embedding or EmbeddingConfig()).embedding_model, 
        hf_token= (config.provided.huggingface or HuggingFaceConfig()).token,
        default_truncate_dim=(config.provided.embedding or EmbeddingConfig()).default_truncate_dim
    )
    vectordb_helper = providers.Factory(QdrantDBHelper, embeddings=local_embedding, host=config.provided.qdrant.host, port=config.provided.qdrant.port)

    database_schema_repo = providers.Factory(DatabaseSchemaRepository,db_manager=db_manager, vectordb=vectordb_helper)
    response_struct_agent = providers.Factory(ResponseStructAgent)
    sql_builder_agent = providers.Factory(SqlBuilderAgent, response_struct_agent=response_struct_agent)

    database_schema_usecase = providers.Factory(DatabaseSchemaUsecase, database_schema_repo=database_schema_repo, sql_builder_agent = sql_builder_agent)