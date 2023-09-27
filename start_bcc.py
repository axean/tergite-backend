from utils.application_starter import BCCApplication

if __name__ == "__main__":
    import settings

    # TODO: Start the rq workers and clean the repositories here as well

    host_name = str(settings.BCC_MACHINE_ROOT_URL).split('://')[1].split(':')[0]
    application = BCCApplication(host=host_name,
                                 port=settings.BCC_PORT)
    application.run()
